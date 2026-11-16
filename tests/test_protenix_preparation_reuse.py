"""CPU transport checks; this does not run Protenix or manufacture predictions."""
import subprocess
import sys
import textwrap


def test_protenix_materialization_reuses_fixed_input_and_updates_query(tmp_path):
    # Keep the optional inference import stub private to a child process. All
    # input conversion, filesystem operations and cache behavior are real.
    code = textwrap.dedent('''
        import importlib.util
        import sys
        import types
        from pathlib import Path
        if importlib.util.find_spec("torch") is None:
            torch = types.ModuleType("torch")
            torch.distributed = types.ModuleType("torch.distributed")
            torch.no_grad = lambda: lambda function: function
            sys.modules["torch"] = torch
            sys.modules["torch.distributed"] = torch.distributed
        from astrolabe.evaluators.wrappers import protenix as wrapper
        from astrolabe.core import global_config
        from astrolabe.core.folding_input import Input, ProteinChain, Template
        from astrolabe.evaluators.preparation_cache import ContentCache
        root = Path(sys.argv[1])
        global_config.reset()
        global_config.init_const("out_dir", root)
        def request(sequence, templates=()):
            chain = ProteinChain(id="A", sequence=sequence, ptms=[],
                                 unpaired_msa="", paired_msa="", templates=list(templates))
            return Input(name="test", chains=[chain], rng_seeds=[1])
        first = wrapper.from_af3_to_protenix(request("ACDE"), "A", use_constraints=False)
        file = Path(first["sequences"][0]["proteinChain"]["unpairedMsaPath"])
        timestamp = file.stat().st_mtime_ns
        second = wrapper.from_af3_to_protenix(request("ACDE"), "A", use_constraints=False)
        assert first == second and file.stat().st_mtime_ns == timestamp
        changed = wrapper.from_af3_to_protenix(request("ACDF"), "A", use_constraints=False)
        assert "ACDF" in file.read_text()
        assert changed["sequences"][0]["proteinChain"]["sequence"] == "ACDF"
        parses = []
        def parse(**kwargs):
            parses.append(kwargs["mmcif_string"])
            return types.SimpleNamespace(mmcif_object=types.SimpleNamespace(
                header={"release_date": "2020-01-01"}, chain_to_seqres={"A": "ACDE"}))
        wrapper._lazy_import_protenix = lambda: None
        wrapper.TemplateParser = types.SimpleNamespace(parse=parse)
        template = Template(mmcif="data_test\\n_entry.id 1ABC\\n", query_to_template_map={0: 0})
        cache = ContentCache()
        one = wrapper.from_af3_to_protenix(request("ACDE", [template]), "A",
                                          use_constraints=False, use_templates=True,
                                          template_cache=cache)
        hit_path = Path(one["sequences"][0]["proteinChain"]["templatesPath"])
        hit_timestamp = hit_path.stat().st_mtime_ns
        wrapper.from_af3_to_protenix(request("ACDF", [template]), "A",
                                    use_constraints=False, use_templates=True,
                                    template_cache=cache)
        assert len(parses) == 1 and hit_path.stat().st_mtime_ns == hit_timestamp
    ''')
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
