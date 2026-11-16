"""Exercise the real evaluator boundary with isolated optional-dependency doubles.

These are coordinator/cache lifetime tests, not native scientific predictions.
"""
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_prediction_custom_ccd_lifetime_and_evaluator_cache_wiring(failure, owned):
    code = textwrap.dedent('''
        import json
        import sys
        import types
        from contextlib import nullcontext
        from types import SimpleNamespace
        import numpy as np

        def module(name, **values):
            result = types.ModuleType(name)
            result.__dict__.update(values)
            sys.modules[name] = result
            if "." in name:
                parent, child = name.rsplit(".", 1)
                if parent in sys.modules:
                    setattr(sys.modules[parent], child, result)
            return result

        module("jax")
        module("alphafold3")
        module("alphafold3.common")
        module("alphafold3.common.folding_input", ProteinChain=object)
        module("alphafold3.constants")
        shared = {"ALA": {"atom": ("base",)}}
        constructions = []
        class Ccd:
            def __init__(self):
                self._dict = shared
                constructions.append(self)
            def items(self): return self._dict.items()
        module("alphafold3.constants.chemical_components", Ccd=Ccd,
               cif_dict=SimpleNamespace(parse_multi_data_cif=json.loads))
        module("alphafold3.model")
        refs = {}
        module("alphafold3.model.features", _REFSTRUCTURE_REFERENCE_CACHE=refs)
        module("alphafold3.data")
        def encode(rows, polymer):
            values = np.array([[ord(char) for char in row] for row in rows])
            return values, np.zeros_like(values)
        msa = module("alphafold3.data.msa_features", extract_msa_features=encode)
        def prepare(**kwargs):
            refs[id(kwargs["ccd"]._dict["ALA"])] = object()
            if sys.argv[1] == "True":
                raise ValueError("preparation failed")
            msa.extract_msa_features([kwargs["fold_input"].sequence, "CCCC"], "protein")
            return [{"current_input": kwargs["fold_input"]}]
        module("alphafold3.data.featurisation", featurise_input=prepare)
        module("astrolabe.evaluators.wrappers.alphafold3", BUCKETS=(128,),
               ResultsForSeed=SimpleNamespace, get_af3_scores=lambda _: None,
               write_outputs=lambda **kwargs: None)
        from astrolabe.evaluators import alphafold3
        alphafold3.allow_all_x_protein_chains_during_featurisation = lambda _: nullcontext()
        fold_input = SimpleNamespace(rng_seeds=[17], name="test", sequence="AAAA",
                                     user_ccd='{"ALA":{"atom":["custom"]}}')
        calls = []
        owned = sys.argv[2] == "True"
        def infer(batch, seed, name):
            assert bool(refs) == owned, "Temporary references survived into inference"
            calls.append((batch, seed, name))
            return []
        runner = SimpleNamespace(run_inference_and_extract_for_seed=infer)
        evaluator = alphafold3.AlphaFold3Evaluator(runner, ".", SimpleNamespace(msa_cache=True))
        # Exercise the real request copy + run_function + prediction boundary;
        # scientific input/output conversion is outside this lifecycle test.
        alphafold3.AlphaFold3Evaluator.prepare_input = lambda self, inputs, config: inputs
        alphafold3.AlphaFold3Evaluator.prepare_output = lambda self, output, inputs, config: output
        try:
            if owned:
                evaluator.evaluate({"test": fold_input})
                fold_input.sequence = "AAAC"
                evaluator.evaluate({"test": fold_input})
                assert len(constructions) == 1
                assert evaluator._msa_rows.encoded_rows == 3
                assert evaluator._msa_rows.hits == 1
            else:
                alphafold3.AlphaFold3Evaluator.predict_structure(fold_input, runner, (128,))
        except ValueError as exc:
            assert sys.argv[1] == "True" and str(exc) == "preparation failed"
        else:
            assert sys.argv[1] == "False" and calls[0][1:] == (17, "test")
        assert shared == {"ALA": {"atom": ("base",)}}
        assert bool(refs) == owned
        evaluator.shutdown()
        assert refs == {}, "Temporary custom CCD left identity-keyed native references alive"
    ''')
    result = subprocess.run([sys.executable, "-c", code, str(failure), str(owned)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
