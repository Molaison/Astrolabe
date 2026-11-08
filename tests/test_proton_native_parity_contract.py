"""No model imports/calls: entrypoint and bounded manifest contracts."""
import copy
import dataclasses
import builtins
import contextlib
import zlib
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "astrolabe/posthoc/proton_parity.py"
spec = importlib.util.spec_from_file_location("parity_contract", MODULE)
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


class ParityContract(unittest.TestCase):
    def test_arm_sources_use_repo_vendor_trees(self):
        repo = Path("/opt/Astrolabe")
        for arm in ("pristine", "clean"):
            self.assertEqual(parity._source_root(repo, arm), (repo / "vendor/proton_pottsmpnn").resolve())
        self.assertEqual(parity._source_root(repo, "legacy"),
                         (repo / "vendor/proton_pottsmpnn_legacy").resolve())

    def original_source(self, root):
        # Actual pinned upstream text, compressed only for portable offline tests.
        # Both files retain upstream licensing in vendor/proton_pottsmpnn.
        fixture = json.loads(zlib.decompress(
            (MODULE.parents[2] / "tests/fixtures/proton_parity_upstream.json.zlib").read_bytes()))
        self.assertEqual(fixture["commit"], parity._UPSTREAM_COMMIT)
        for relative, entry in fixture["files"].items():
            self.assertEqual(entry["upstream_git_blob"], parity._UPSTREAM_BLOBS[relative])
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(entry["content"].encode("utf-8"))
        return root

    def test_pins_match_verified_upstream_not_patched_release(self):
        repo = MODULE.parents[2]
        manifest = json.loads((repo / "vendor/proton_pottsmpnn/SOURCE_PROVENANCE.json").read_text())
        self.assertEqual(manifest["commit"], parity._UPSTREAM_COMMIT)
        for relative, blob in parity._UPSTREAM_BLOBS.items():
            entry = next(f for f in manifest["files"] if f["path"] == relative)
            self.assertEqual(entry["upstream_git_blob"], blob)
            content = (repo / "vendor/proton_pottsmpnn" / relative).read_bytes()
            patched_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            self.assertNotEqual(patched_blob, blob)

    def test_actual_upstream_passes_and_each_file_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.original_source(Path(directory))
            for arm in ("pristine", "clean"):
                with self.subTest(arm=arm):
                    result = parity._validate_source(root, arm)
                    self.assertEqual(result["kind"], "pinned_upstream")
                    self.assertEqual(result["upstream_commit"], parity._UPSTREAM_COMMIT)
                    self.assertEqual({k: v["git_blob"] for k, v in result["files"].items()}, parity._UPSTREAM_BLOBS)
            # Even a manifest claiming the right commit cannot bless changed bytes.
            (root / "SOURCE_PROVENANCE.json").write_text(json.dumps({"commit": parity._UPSTREAM_COMMIT}))
            for relative in parity._UPSTREAM_BLOBS:
                path = root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n# local change\n")
                try:
                    for arm in ("pristine", "clean"):
                        with self.subTest(changed=relative, arm=arm), self.assertRaisesRegex(ValueError, "--source-root"):
                            parity._validate_source(root, arm)
                finally:
                    path.write_bytes(original)

    def test_patched_default_rejected_before_imports_or_asset_access(self):
        repo = MODULE.parents[2]
        before = {relative: (repo / "vendor/proton_pottsmpnn" / relative).read_bytes()
                  for relative in parity._UPSTREAM_BLOBS}
        original_import = builtins.__import__
        attempted = []

        def no_heavy_import(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "torch", "atomworks", "mpnn", "foundry"}:
                attempted.append(name)
                raise AssertionError("Source rejection must precede heavy imports")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.json"
            inputs.write_text(json.dumps(self.manifest()))
            for arm in ("pristine", "clean"):
                with self.subTest(arm=arm), mock.patch("builtins.__import__", side_effect=no_heavy_import), contextlib.redirect_stdout(io.StringIO()):
                    code = parity.main(["--repo-root", str(repo), "--arm", arm,
                                        "--checkpoint", str(root / "missing.ckpt"),
                                        "--hbplus", str(root / "missing-hbplus"),
                                        "--inputs", str(inputs), "--output", str(root / arm)])
                self.assertEqual(code, 1)
                report = json.loads((root / arm / "report.json").read_text())
                self.assertIn("requires exact upstream", report["traceback"])
                self.assertIn("--source-root", report["traceback"])
                self.assertTrue(all(n == 0 for n in report["budgets"].values()))
        self.assertEqual(attempted, [])
        self.assertEqual(before, {relative: (repo / "vendor/proton_pottsmpnn" / relative).read_bytes()
                                  for relative in parity._UPSTREAM_BLOBS})

    def test_explicit_upstream_source_reaches_asset_check_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.original_source(root / "original")
            inputs = root / "inputs.json"
            inputs.write_text(json.dumps(self.manifest()))
            with contextlib.redirect_stdout(io.StringIO()):
                code = parity.main(["--repo-root", str(MODULE.parents[2]), "--arm", "clean",
                                    "--source-root", str(source), "--checkpoint", str(root / "missing.ckpt"),
                                    "--hbplus", str(root / "missing-hbplus"), "--inputs", str(inputs),
                                    "--output", str(root / "out")])
            report = json.loads((root / "out/report.json").read_text())
            self.assertEqual(code, 1)  # Deliberately missing asset prevents any model import.
            self.assertEqual(report["source_provenance"]["kind"], "pinned_upstream")
            self.assertEqual(report["source_root"], str(source))
            self.assertIn("missing.ckpt", report["traceback"])
            self.assertNotIn("requires exact upstream", report["traceback"])
            self.assertTrue(all(n == 0 for n in report["budgets"].values()))

    def test_legacy_source_records_actual_bytes_without_claiming_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root = self.original_source(parity._source_root(repo, "legacy"))
            relative = next(iter(parity._UPSTREAM_BLOBS))
            path = root / relative
            path.write_bytes(path.read_bytes() + b"\n# legacy extension\n")
            result = parity._validate_source(root, "legacy")
            self.assertEqual(result["root"], str(root))
            self.assertEqual(result["kind"], "legacy_snapshot")
            self.assertIsNone(result["upstream_commit"])
            self.assertEqual(result["files"][relative]["sha256"], parity._sha(path))
            self.assertNotEqual(result["files"][relative]["git_blob"], parity._UPSTREAM_BLOBS[relative])

    def manifest(self):
        return {"schema": "proton_native_parity_v1", "contexts": ["full_binder_mask", "original_restricted_mask"],
                "items": [{"item_id": "fixture", "binder_sequence": "AHH", "mutable_positions": [1, 2],
                           "fixed_positions": [0], "histidine_positions": [1, 2],
                           "centers": [{"res_id": p, "protonation_type": "HIS-S"} for p in (2, 3)]}],
                "criteria": {"shared": dict(seed_source="inverse", selective_source="potts", selective=True,
                              combined_lambda=.3, max_mutations=3, neighbour_k=16, dep_map={"HIS-S": ["HIS-P"]}),
                             "deterministic": dict(method="block_descent", backend="potts", temperature=0., block_size=2),
                             "stochastic": dict(method="autoregressive", backend="mpnn", temperature=.2)}}

    def test_standalone_import_and_help_have_no_heavy_imports(self):
        code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('p',sys.argv[1]); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert not any(n.split('.')[0] in ('astrolabe','torch','mpnn','foundry','atomworks') for n in sys.modules); "
                "m.main(['--help'])")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(MODULE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--arm", result.stdout)

    def test_one_output_per_method_and_original_manifest_unchanged(self):
        manifest = self.manifest()
        original = copy.deepcopy(manifest)
        parity._validate_manifest(manifest)
        for params in parity._criteria(manifest, manifest["items"][0]).values():
            self.assertEqual(params["samples_per_site"], 1)
            self.assertNotIn("num_designs", params)
            self.assertEqual(params["explicit_centers"], original["items"][0]["centers"])
        self.assertEqual(manifest, original)

    def test_sign_flip_and_extra_samples_require_decision(self):
        for field, value in (("dep_map", {"HIS-P": ["HIS-S"]}), ("samples_per_site", 4)):
            manifest = self.manifest()
            manifest["criteria"]["shared"][field] = value
            with self.assertRaisesRegex(ValueError, "NEED_DECISION"):
                parity._validate_manifest(manifest)

    def test_unsafe_partition_rejected(self):
        manifest = self.manifest()
        manifest["items"][0]["fixed_positions"].append(1)
        with self.assertRaises(ValueError):
            parity._validate_manifest(manifest)

    def test_raw_metadata_serialization_is_lossless(self):
        raw = {"canonical_sequence": "AHH", "extended_tokens": ["ALA", "HIS-S", "HIS-P"],
               "energy_trajectory": [{"selective_energy": -1.5}]}
        self.assertEqual(json.loads(json.dumps(parity._jsonable(raw))), raw)

    def test_nested_nonfinite_annotations_preserved_in_strict_json(self):
        class ArrayLike:
            def tolist(self):
                return [float("nan"), {"missing": float("inf")}]

        class TensorLike:
            def detach(self):
                return self

            def cpu(self):
                return ArrayLike()

        raw = {"annotations": [ArrayLike(), TensorLike(), {"missing": float("-inf")}], "finite": 1.25}
        expected = {"annotations": [[{"nonfinite": "nan"}, {"missing": {"nonfinite": "inf"}}],
                                    [{"nonfinite": "nan"}, {"missing": {"nonfinite": "inf"}}],
                                    {"missing": {"nonfinite": "-inf"}}], "finite": 1.25}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            parity._write(path, raw)
            result = json.loads(path.read_text(), parse_constant=lambda token: self.fail(token))
        self.assertEqual(result, expected)

    def test_nonfinite_scientific_energies_fail_before_sanitization(self):
        @dataclasses.dataclass
        class Output:
            final_potts_energy: float

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "Nonfinite scientific energy at states.SS"):
                parity._require_finite_energy(value, "states.SS")
            for raw in ({"potts_energy": value}, {"selective_energies": [0., value]},
                        {"energy_trajectory": [{"selective_energy": value}]},
                        {"global_protonation_dH": value}, Output(value)):
                with self.assertRaisesRegex(ValueError, "Nonfinite scientific energy"):
                    parity._check_scientific_energies(raw)
        parity._check_scientific_energies({"potts_energy": 0., "selective_energy": -1.5,
                                           "global_protonation_dH": None})

    def test_annotation_name_order_is_not_a_token_value_difference(self):
        output = {"native_fields": {"canonical_sequence": "HH", "extended_tokens": ["HIS-S", "HIS-S"]}}
        case = {"item_id": "fixture", "mode": "full_binder_mask", "status": "complete",
                "context_arrays": {k: "same" for k in ("S_native", "E_idx", "etab")},
                "context_tokens_sha256": "raw_order_a", "token_to_idx": {"HIS-S": 0},
                "states": {"SS": 0.}, "single_center_S_minus_P": [-1., -2.],
                "context_token_annotation": {"annotation_categories": ["chain_id", "res_id"],
                    "tokens": {"chain_id": ["A", "A"], "res_id": [1, 2]},
                    "atom_count": 2, "token_count": 2},
                "native_runs": {k: {"outputs": [output]} for k in ("deterministic", "stochastic")}}
        first = {"arm": "pristine", "manifest_sha256": "same", "checkpoint": {"sha256": "same"},
                 "hbplus": {"sha256": "same"}, "cases": [case]}
        second = copy.deepcopy(first)
        second["arm"] = "clean"
        right = second["cases"][0]
        right["context_tokens_sha256"] = "raw_order_b"
        right["context_token_annotation"]["annotation_categories"].reverse()
        result = parity.compare_reports([first, second])[0]
        self.assertFalse(result["raw_annotation_hash_exact"])
        self.assertTrue(result["tokens_exact"])
        self.assertEqual(result["status"], "match")
        right["context_token_annotation"]["tokens"]["res_id"][1] = 3
        result = parity.compare_reports([first, second])[0]
        self.assertFalse(result["tokens_exact"])
        self.assertEqual(result["status"], "difference")


if __name__ == "__main__":
    unittest.main()
