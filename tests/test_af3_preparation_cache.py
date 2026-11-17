from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

if "astrolabe.af3_preparation" in sys.modules:
    _PREPARATION = sys.modules["astrolabe.af3_preparation"]
else:
    _SPEC = importlib.util.spec_from_file_location(
        "astrolabe.af3_preparation", Path(__file__).parents[1] / "astrolabe" / "af3_preparation.py")
    _PREPARATION = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _PREPARATION
    _SPEC.loader.exec_module(_PREPARATION)

AF3PreparationCache = _PREPARATION.AF3PreparationCache

try:
    from alphafold3.data import msa_features
    from alphafold3.constants import mmcif_names
except ImportError:
    msa_features = None


@unittest.skipIf(msa_features is None, "native AF3 MSA extraction is not installed")
class MsaDeltaTests(unittest.TestCase):
    def test_mutated_query_reuses_only_unchanged_rows_and_returns_owned_arrays(self):
        original = msa_features.extract_msa_features
        rows_computed = []
        def extract(*, msa_sequences, chain_poly_type):
            rows_computed.append(tuple(msa_sequences))
            return original(msa_sequences, chain_poly_type)
        cache = AF3PreparationCache()
        tail = ["ACDkEFG", "AC-EFG"]
        for query in ("ACDEFG", "ACKEFG", "ACDEFG"):
            rows = [query, *tail]
            actual = cache.extract_msa_features(
                extract, msa_sequences=rows, chain_poly_type=mmcif_names.PROTEIN_CHAIN)
            expected = original(rows, mmcif_names.PROTEIN_CHAIN)
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)
                a[:] = -99
        self.assertEqual(rows_computed, [("ACDEFG",), tuple(tail), ("ACKEFG",), ("ACDEFG",)])
        self.assertEqual(cache.statistics["msa_tail_hits"], 2)

    def test_homologue_edits_and_polymer_types_invalidate(self):
        cache = AF3PreparationCache()
        for rows, chain_type in [
            (["ACGU", "ACGU"], mmcif_names.RNA_CHAIN),
            (["ACGU", "A-GU"], mmcif_names.RNA_CHAIN),
            (["ACGU", "A-GU"], mmcif_names.PROTEIN_CHAIN),
        ]:
            actual = cache.extract_msa_features(
                msa_features.extract_msa_features, msa_sequences=rows, chain_poly_type=chain_type)
            expected = msa_features.extract_msa_features(rows, chain_type)
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(cache.statistics["msa_tail_misses"], 3)

    def test_invalid_alignment_still_fails_and_cache_is_bounded(self):
        cache = AF3PreparationCache(max_bytes=128, max_entries=1)
        original = msa_features.extract_msa_features
        for query in ("ACDEFG", "ACKEFG"):
            cache.extract_msa_features(
                original, msa_sequences=[query, query], chain_poly_type=mmcif_names.PROTEIN_CHAIN)
        self.assertLessEqual(cache.statistics["entries"], 1)
        self.assertLessEqual(cache.statistics["bytes"], 128)
        with self.assertRaises(ValueError):
            cache.extract_msa_features(
                original, msa_sequences=["ACDEFGH", "ACDEFG"],
                chain_poly_type=mmcif_names.PROTEIN_CHAIN)
        with self.assertRaises(ValueError):
            cache.extract_msa_features(
                original, msa_sequences=["ACDEFG", "AC$EFG"],
                chain_poly_type=mmcif_names.PROTEIN_CHAIN)
        cache.clear()
        self.assertEqual(cache.statistics["bytes"], 0)


@dataclass
class Protein:
    id: str
    sequence: str
    templates: tuple = ()
    ptms: tuple = ()


@dataclass(frozen=True)
class Padding:
    num_tokens: int = 12
    num_templates: int = 4


class TemplateReuseTests(unittest.TestCase):
    def make_arguments(self, sequences=("ACDEFG", "CKDEFG")):
        chains = [Protein("A", sequences[0]), Protein("B", sequences[1])]
        return dict(
            all_tokens=SimpleNamespace(
                chain_id=np.array(["A"] * 6 + ["B"] * 6, dtype=object),
                chain_type=np.array(["polypeptide(L)"] * 12, dtype=object),
                res_name=np.array(list("".join(sequences)), dtype=object),
                res_id=np.tile(np.arange(1, 7), 2),
                atom_name=np.array(["CA"] * 12, dtype=object)),
            standard_token_idxs=np.arange(12),
            padding_shapes=Padding(),
            fold_input=SimpleNamespace(chains=chains, bonded_atom_pairs=None, user_ccd=None),
            max_templates=4, logging_name="request")

    def test_sequence_delta_and_name_reuse_but_entity_mapping_and_templates_invalidate(self):
        cache = AF3PreparationCache()
        calls = []
        def native(**arguments):
            calls.append(arguments)
            return {"features": np.array([len(calls)], dtype=float)}
        first = cache.template_features(native, self.make_arguments(), protein_type=Protein)
        first["features"][:] = -1
        mutant = self.make_arguments(("DCDEFG", "CKDEFG"))
        mutant["logging_name"] = "another seed"
        reused = cache.template_features(native, mutant, protein_type=Protein)
        np.testing.assert_array_equal(reused["features"], [1])
        collision = self.make_arguments(("CKDEFG", "CKDEFG"))
        cache.template_features(native, collision, protein_type=Protein)
        mutant["fold_input"].chains[0].templates = (
            SimpleNamespace(mmcif="data_changed", query_to_template_map={0: 2}),)
        cache.template_features(native, mutant, protein_type=Protein)
        self.assertEqual(len(calls), 3)

    def test_chemical_topology_and_unrecognized_arguments_fall_back(self):
        cache = AF3PreparationCache()
        calls = []
        def native(**arguments):
            calls.append(arguments)
            return np.ones(4)
        for field in ("user_ccd", "bonded_atom_pairs"):
            args = self.make_arguments()
            setattr(args["fold_input"], field, "present")
            for _ in range(2):
                cache.template_features(native, args, protein_type=Protein)
        modified = self.make_arguments()
        modified["fold_input"].chains[0].ptms = (("MSE", 1),)
        cache.template_features(native, modified, protein_type=Protein)
        changed_api = self.make_arguments()
        changed_api["new_native_option"] = True
        cache.template_features(native, changed_api, protein_type=Protein)
        self.assertEqual(len(calls), 6)
        self.assertEqual(cache.statistics["entries"], 0)


if __name__ == "__main__":
    unittest.main()
