from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from astrolabe.posthoc import refold_receipts
from astrolabe.posthoc.db import open_db, upsert_records
from astrolabe.posthoc.refold import export_refold_manifest


_REAL_SELECTED_ASSET_RECORD = refold_receipts.selected_asset_record
_REAL_NUMERIC_SOURCE_RECORD = refold_receipts.numeric_source_record
_REAL_SLOT_COMPLETE = refold_receipts.RefoldReceiptSlot.complete


@pytest.fixture(autouse=True)
def _fast_posthoc_receipt_identity(monkeypatch):
    """Keep fake-model tests bounded by real paid assets."""

    def selected_asset_record(*, backend, selected_config):
        payload = json.dumps(
            selected_config,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return {
            "schema": "test.refold-assets/v1",
            "complete": True,
            "components": [
                {
                    "role": f"evaluator:{backend}",
                    "type": backend,
                    "state": "hashed",
                    "detail": None,
                }
            ],
            "files": [
                {
                    "role": f"evaluator:{backend}",
                    "path": f"test://{backend}",
                    "sha256": digest,
                    "size_bytes": len(payload),
                }
            ],
        }

    monkeypatch.setattr(
        refold_receipts,
        "selected_asset_record",
        selected_asset_record,
    )
    monkeypatch.setattr(
        refold_receipts,
        "numeric_source_record",
        lambda backend, repo_root=None: {
            "schema": "test.refold-source/v1",
            "backend": backend,
            "files": [],
        },
    )

    def complete_with_test_outputs(self, *, identity, output_roots):
        if self.backend == "af3":
            marker = (
                self.run_dir
                / "raw_model_results"
                / "alphafold3"
                / f"step_{self.step:03d}_{self.oligo}"
                / f"step_{self.step:03d}_{self.oligo}_summary_confidences.json"
            )
            structure = (
                self.run_dir
                / "structures"
                / f"step_{self.step:03d}_{self.oligo}_model.cif"
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}\n", encoding="utf-8")
            structure.parent.mkdir(parents=True, exist_ok=True)
            structure.write_text("data_test\n#\n", encoding="utf-8")
        else:
            base = (
                self.run_dir
                / "raw_model_results"
                / "protenix"
                / f"step_{self.step:03d}"
                / self.oligo
                / "seed_1"
                / "predictions"
            )
            base.mkdir(parents=True, exist_ok=True)
            (base / "test_summary_confidence_sample_0.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
        return _REAL_SLOT_COMPLETE(
            self,
            identity=identity,
            output_roots=output_roots,
        )

    monkeypatch.setattr(
        refold_receipts.RefoldReceiptSlot,
        "complete",
        complete_with_test_outputs,
    )


def _write_source_receipt(
    item: dict,
    *,
    backend: str = "af3",
    real_assets: bool = False,
) -> None:
    from omegaconf import OmegaConf

    from astrolabe.posthoc import refold

    source_run_dir = Path(item["src_run_dir"]).resolve()
    config_path = source_run_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)
    selected_config = refold._load_eval_subcfg(
        cfg,
        "alphafold3" if backend == "af3" else "protenix",
    )
    runtime_config = refold._load_refold_runtime_overrides(cfg)
    input_context = refold._load_refold_input_context(cfg, item=item)
    homologies = refold._resolve_homology_value(
        v=item.get("homologies") or {},
        relpath_base=None,
        path_maps=(),
        require_abs=True,
    )
    selected_asset_record = (
        _REAL_SELECTED_ASSET_RECORD
        if real_assets
        else refold_receipts.selected_asset_record
    )
    numeric_source_record = (
        _REAL_NUMERIC_SOURCE_RECORD
        if real_assets
        else refold_receipts.numeric_source_record
    )
    identity = refold_receipts.build_scientific_identity(
        backend=backend,
        item={**item, "_receipt_seed": int(cfg.seed)},
        config_path=config_path,
        selected_config=selected_config,
        compiled_input_context=input_context,
        runtime_config=runtime_config,
        homologies=homologies,
        asset_record=selected_asset_record(
            backend=backend,
            selected_config=selected_config,
        ),
        numeric_source=numeric_source_record(
            backend,
            repo_root=Path(__file__).resolve().parents[1],
        ),
    )
    slot = refold_receipts.RefoldReceiptSlot(
        source_run_dir,
        backend,
        int(item["step"]),
        str(item["oligo"]),
    )
    with slot.lock():
        slot.issue(identity, origin="inference")
        _REAL_SLOT_COMPLETE(
            slot,
            identity=identity,
            output_roots=refold._backend_output_roots(
                source_run_dir,
                backend=backend,
                step=int(item["step"]),
                oligo=str(item["oligo"]),
            ),
        )


def _posthoc_af3_case(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = tmp_path / "source" / "task_a" / "001_run"
    _write_refold_af3_config(source / "config.yaml")
    item = {
        "task": "task_a",
        "run": "001_run",
        "step": 0,
        "oligo": "A",
        "src_run_dir": str(source),
        "src_config": str(source / "config.yaml"),
        "seqs": {"A": "AAAA"},
        "types": {"A": "protein"},
        "homologies": {"A": None},
        "needed_evaluators": ["af3"],
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(item) + "\n", encoding="utf-8")
    return manifest, tmp_path / "refold_runs", item


def _install_fake_af3(monkeypatch) -> dict[str, int]:
    from types import SimpleNamespace

    calls = {"construct": 0, "evaluate": 0}

    class FakeAF3:
        def evaluate(self, _inputs, runtime_config=None):
            calls["evaluate"] += 1
            return SimpleNamespace(
                iptm={"A": [0.8]},
                ptm={"A": [0.7]},
                plddt={"A": [0.9]},
            )

    def construct(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["A"]
        return {"Alphafold3": FakeAF3()}

    monkeypatch.setattr(
        "astrolabe.posthoc.refold._init_refold_global_config",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._build_protomers",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"A": "input"},
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_evaluators",
        construct,
    )
    return calls


def test_refold_from_db_propagates_resource_group_summary(monkeypatch, tmp_path: Path) -> None:
    import astrolabe.posthoc.refold as refold_mod

    def _export_refold_manifest(**kwargs):
        Path(kwargs["out_manifest"]).write_text("", encoding="utf-8")
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "preset",
            "selector": "",
            "preset": "preset",
            "filter_expr": "",
            "written": 2,
        }

    def _run_refold(**_kwargs):
        return {
            "processed": 2,
            "skipped": 1,
            "failed": 0,
            "ran_af3": 1,
            "ran_protenix": 1,
            "reused_af3": 1,
            "reused_protenix": 0,
            "groups": 2,
            "resource_groups": 1,
            "evaluator_cache_hits": 1,
            "evaluator_cache_misses": 1,
            "failure_report": None,
        }

    monkeypatch.setattr(refold_mod, "export_refold_manifest", _export_refold_manifest)
    monkeypatch.setattr(refold_mod, "run_refold", _run_refold)

    summary = refold_mod.refold_from_db(
        db_path=tmp_path / "scores.sqlite",
        preset_path=tmp_path / "preset.yaml",
        dst_dir=tmp_path / "refold",
        dry_run=True,
    )

    assert summary["groups"] == 2
    assert summary["resource_groups"] == 1
    assert summary["evaluator_cache_hits"] == 1
    assert summary["evaluator_cache_misses"] == 1
    assert summary["reused_af3"] == 1


def _seed_record_with_config(*, db_path: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "homologies:",
                "  A: null",
                "  B: null",
                "types:",
                "  A: protein",
                "  B: protein",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    conn = open_db(db_path)
    try:
        upsert_records(
            conn,
            [
                {
                    "run_dir": run_dir,
                    "task": "task_a",
                    "run": "001_run",
                    "step": 0,
                    "oligo": "AB",
                    "structure_path": "",
                    "metrics": {
                        "stage": "default",
                        "weighted_score": 1.0,
                        "pA_sequence": "AAAA",
                        "pB_sequence": "BBBB",
                    },
                    "scores_mtime_ns": 1,
                    "scores_size": 10,
                }
            ],
        )
    finally:
        conn.close()


def test_export_refold_manifest_includes_rows_blocked_only_by_missing_preset_metrics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_record_with_config(db_path=db_path, run_dir=run_dir)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["src_run_dir"] == str(run_dir.resolve())


def test_export_refold_manifest_includes_rows_blocked_only_by_missing_filter_metrics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_record_with_config(db_path=db_path, run_dir=run_dir)

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        filter_expr="protenix_iptm > 0.7",
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["selector_kind"] == "dsl"
    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["src_run_dir"] == str(run_dir.resolve())


def _seed_refold_record(
    *,
    db_path: Path,
    run_dir: Path,
    metrics: dict[str, object],
    stage_alias: str = "StrictPX",
    stage_type: str = "protenix",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "homologies:",
                "  A: null",
                "  B: null",
                "types:",
                "  A: protein",
                "  B: protein",
                "stages:",
                f"  default:\n    evaluator: {stage_alias}",
                "evaluator:",
                "  evaluator_terms:",
                f"    {stage_alias}:\n      _type: {stage_type}",
                "    AF3Alias:\n      _type: alphafold3",
                "  protenix:\n    repo_path: /tmp/protenix",
                "  alphafold3:\n    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    conn = open_db(db_path)
    try:
        upsert_records(
            conn,
            [
                {
                    "run_dir": run_dir,
                    "task": "task_a",
                    "run": "001_run",
                    "step": 0,
                    "oligo": "AB",
                    "structure_path": "",
                    "metrics": metrics,
                    "scores_mtime_ns": 1,
                    "scores_size": 10,
                }
            ],
        )
    finally:
        conn.close()


def test_export_refold_manifest_skips_rows_that_already_pass_selector(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_refold_record(
        db_path=db_path,
        run_dir=run_dir,
        metrics={
            "stage": "default",
            "weighted_score": 1.0,
            "pA_sequence": "AAAA",
            "pB_sequence": "BBBB",
            "protenix_iptm": 0.91,
        },
    )
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["records_passed"] == 1
    assert summary["written"] == 0
    assert manifest_path.read_text(encoding="utf-8") == ""


def test_export_refold_manifest_marks_needed_evaluators_from_missing_metrics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_refold_record(
        db_path=db_path,
        run_dir=run_dir,
        metrics={
            "stage": "default",
            "weighted_score": 1.0,
            "pA_sequence": "AAAA",
            "pB_sequence": "BBBB",
        },
    )
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["needed_evaluators"] == ["protenix"]


def test_run_refold_only_requires_backend_configs_that_manifest_needs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    src_run_dir.mkdir(parents=True, exist_ok=True)
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    item = {
        "task": "task_a",
        "run": "001_run",
        "step": 0,
        "oligo": "AB",
        "src_run_dir": str(src_run_dir),
        "src_config": str(src_run_dir / "config.yaml"),
        "seqs": {"A": "AAAA", "B": "BBBB"},
        "types": {"A": "protein", "B": "protein"},
        "homologies": {"A": None, "B": None},
        "needed_evaluators": ["af3"],
    }
    manifest_path.write_text(
        json.dumps(item)
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3", "protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 1
    assert summary["ran_protenix"] == 0


def test_run_refold_restores_cyclization_and_nonstandard_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.core import global_config
    from astrolabe.posthoc.refold import run_refold

    source_dir = tmp_path / "source" / "task_a" / "001_run"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "A",
                "src_run_dir": str(source_dir),
                "src_config": str(source_dir / "config.yaml"),
                "seqs": {"A": "A[S5]AAAA[R8]A"},
                "types": {"A": "protein"},
                "homologies": {"A": None},
                "needed_evaluators": ["af3"],
                "nonstandard_residues": {
                    "aliases": {
                        "S5": {
                            "ccd_id": "AS5",
                            "parent_aa": "A",
                            "atom_names": [
                                "SC5",
                                "SC4",
                                "SC3",
                                "SC2",
                                "SC1",
                                "CA",
                                "N",
                                "CB",
                                "C",
                                "O",
                                "OXT",
                            ],
                            "source": {
                                "format": "smiles",
                                "value": "C=CCCC[C@](N)(C)C(=O)O",
                            },
                        },
                        "R8": {
                            "ccd_id": "VR8",
                            "parent_aa": "A",
                            "atom_names": [
                                "SC8",
                                "SC7",
                                "SC6",
                                "SC5",
                                "SC4",
                                "SC3",
                                "SC2",
                                "SC1",
                                "CA",
                                "N",
                                "CB",
                                "C",
                                "O",
                                "OXT",
                            ],
                            "source": {
                                "format": "smiles",
                                "value": "C=CCCCCCC[C@@](N)(C)C(=O)O",
                            },
                        },
                    }
                },
                "cyclization": {
                    "bonds": [
                        {
                            "a": {
                                "proto": "A",
                                "pos": 2,
                                "aa": "A",
                                "atom": "SC5",
                                "residue": "S5",
                            },
                            "b": {
                                "proto": "A",
                                "pos": 7,
                                "aa": "A",
                                "atom": "SC8",
                                "residue": "R8",
                            },
                            "order": "single",
                            "target_distance": 1.54,
                            "signal_mode": "both",
                        }
                    ],
                    "encoding": {"scope": "all_bonds"},
                    "signal_mode": "bond_only",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _construct_standard_input(oligo_names, seed):
        from astrolabe.utils.base import construct_standard_input

        inputs = construct_standard_input(oligo_names, seed)
        fold_input = inputs["A"]
        captured["bonded_atom_pairs"] = fold_input.bonded_atom_pairs
        captured["cyclic_relative_patches"] = fold_input.cyclic_relative_patches
        captured["user_ccd"] = fold_input.user_ccd
        captured["ptm_dict"] = dict(global_config.protomers["A"].ptm_dict)
        return inputs

    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        _construct_standard_input,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"A": [0.81]},
                ptm={"A": [0.72]},
                plddt={"A": [0.9]},
            )

    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_evaluators",
        lambda _cfg, _out_root, oligo_names: {"Alphafold3": _FakeAF3Evaluator()},
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 1
    assert captured["bonded_atom_pairs"] == [(("A", 2, "SC5"), ("A", 7, "SC8"))]
    assert captured["cyclic_relative_patches"]
    assert "data_AS5" in str(captured["user_ccd"])
    assert captured["ptm_dict"] == {1: "AS5", 6: "VR8"}


def test_run_refold_reuses_source_af3_outputs_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    af3_dir = src_run_dir / "raw_model_results" / "alphafold3" / "step_000_AB"
    af3_dir.mkdir(parents=True, exist_ok=True)
    (af3_dir / "step_000_AB_summary_confidences.json").write_text("{}\n", encoding="utf-8")
    (src_run_dir / "structures").mkdir(parents=True, exist_ok=True)
    (src_run_dir / "structures" / "step_000_AB_model.cif").write_text("data\n", encoding="utf-8")
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    item = {
        "task": "task_a",
        "run": "001_run",
        "step": 0,
        "oligo": "AB",
        "src_run_dir": str(src_run_dir),
        "src_config": str(src_run_dir / "config.yaml"),
        "seqs": {"A": "AAAA", "B": "BBBB"},
        "types": {"A": "protein", "B": "protein"},
        "homologies": {"A": None, "B": None},
        "needed_evaluators": ["af3"],
    }
    _write_source_receipt(item)
    manifest_path.write_text(
        json.dumps(item)
        + "\n",
        encoding="utf-8",
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("construct_evaluators should not be called when source outputs can be reused")

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _boom)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    dst_run_dir = tmp_path / "refold_runs" / "task_a" / "001_run"
    assert summary["failed"] == 0
    assert summary["reused_af3"] == 1
    assert summary["ran_af3"] == 0
    assert (dst_run_dir / "raw_model_results" / "alphafold3" / "step_000_AB" / "step_000_AB_summary_confidences.json").exists()
    assert (dst_run_dir / "structures" / "step_000_AB_model.cif").exists()


def test_run_refold_does_not_reuse_source_outputs_when_external_config_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    af3_dir = src_run_dir / "raw_model_results" / "alphafold3" / "step_000_AB"
    af3_dir.mkdir(parents=True, exist_ok=True)
    (af3_dir / "step_000_AB_summary_confidences.json").write_text("{}\n", encoding="utf-8")
    (src_run_dir / "structures").mkdir(parents=True, exist_ok=True)
    (src_run_dir / "structures" / "step_000_AB_model.cif").write_text("data\n", encoding="utf-8")
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3_a",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    external_cfg = tmp_path / "external.yaml"
    external_cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3_b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["af3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        config_yaml=external_cfg,
    )

    assert summary["failed"] == 0
    assert summary["reused_af3"] == 0
    assert summary["ran_af3"] == 1
    assert calls["construct"] == 1


def test_source_receipt_identity_conflict_runs_target_without_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    source = Path(item["src_run_dir"])
    marker = (
        source
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text('{"source": "completed"}\n', encoding="utf-8")
    structure = source / "structures" / "step_000_A_model.cif"
    structure.parent.mkdir(parents=True)
    structure.write_text("data_source\n#\n", encoding="utf-8")
    _write_source_receipt(item)
    item["seqs"]["A"] = "CCCC"
    manifest.write_text(json.dumps(item) + "\n", encoding="utf-8")
    calls = _install_fake_af3(monkeypatch)

    summary = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["reused_af3"] == 0
    assert summary["ran_af3"] == 1
    assert calls == {"construct": 1, "evaluate": 1}
    with (out_root / "task_a" / "001_run" / "scores.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))
    assert "source receipt did not prove the target scientific identity" in row["note"]


def test_source_issued_receipt_without_terminal_proof_stops_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    source = Path(item["src_run_dir"])
    marker = (
        source
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text('{"source": "unknown"}\n', encoding="utf-8")
    structure = source / "structures" / "step_000_A_model.cif"
    structure.parent.mkdir(parents=True)
    structure.write_text("data_unknown\n#\n", encoding="utf-8")
    _write_source_receipt(item)
    receipt_path = (
        source
        / ".framework"
        / "refold"
        / "af3"
        / "step_000_A"
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "issued"
    receipt.pop("outputs")
    receipt.pop("completed_unix_s")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    calls = _install_fake_af3(monkeypatch)

    summary = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    target = out_root / "task_a" / "001_run"
    assert summary["failed"] == 1
    assert summary["ran_af3"] == 0
    assert calls == {"construct": 0, "evaluate": 0}
    assert not (
        target
        / ".framework"
        / "refold"
        / "af3"
        / "step_000_A"
        / "receipt.json"
    ).exists()
    failure = json.loads(
        Path(summary["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert "issued without a terminal receipt" in failure["error"]


def test_source_slot_lock_contention_abandons_reuse_and_runs_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    source = Path(item["src_run_dir"])
    marker = (
        source
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text('{"source": "completed"}\n', encoding="utf-8")
    structure = source / "structures" / "step_000_A_model.cif"
    structure.parent.mkdir(parents=True)
    structure.write_text("data_source\n#\n", encoding="utf-8")
    _write_source_receipt(item)
    slot = refold_receipts.RefoldReceiptSlot(source, "af3", 0, "A")
    ready = threading.Event()
    release = threading.Event()

    def hold_source_slot() -> None:
        with slot.lock():
            ready.set()
            release.wait(10)

    owner = threading.Thread(target=hold_source_slot)
    owner.start()
    assert ready.wait(5)
    calls = _install_fake_af3(monkeypatch)
    try:
        summary = run_refold(
            manifest_jsonl=manifest,
            out_root=out_root,
            evaluators=["af3"],
        )
    finally:
        release.set()
        owner.join(timeout=5)

    assert not owner.is_alive()
    assert summary["failed"] == 0
    assert summary["reused_af3"] == 0
    assert summary["ran_af3"] == 1
    assert calls == {"construct": 1, "evaluate": 1}
    with (out_root / "task_a" / "001_run" / "scores.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))
    assert "source receipt slot could not be locked" in row["note"]


def test_run_refold_protenix_propagates_binding_site_from_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    src_run_dir.mkdir(parents=True, exist_ok=True)
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5,6",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["protenix"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"AB": "input"},
    )
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_kwargs: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )

    captured: dict[str, object] = {}

    class _FakeProtenixEvaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return None

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        from astrolabe.posthoc import refold as refold_module

        assert oligo_names == ["AB"]
        captured["binding_site"] = refold_module.global_config.binding_site
        captured["protenix_constraint"] = refold_module.global_config.protenix_constraint
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_protenix"] == 1
    assert captured["binding_site"] == {
        "AB": {"binder": "A", "target": "B:5,6", "mode": "pocket"}
    }
    assert captured["protenix_constraint"] is None


def test_run_refold_does_not_reuse_source_protenix_outputs_when_binding_site_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    pred_dir = (
        src_run_dir
        / "raw_model_results"
        / "protenix"
        / "step_000"
        / "AB"
        / "seed_1"
        / "predictions"
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "AB_seed_1_summary_confidence_sample_0.json").write_text("{}\n", encoding="utf-8")
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5,6",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    external_cfg = tmp_path / "external.yaml"
    external_cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["protenix"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"AB": "input"},
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}

    class _FakeProtenixEvaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return None

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
        config_yaml=external_cfg,
    )

    assert summary["failed"] == 0
    assert summary["reused_protenix"] == 0
    assert summary["ran_protenix"] == 1
    assert calls["construct"] == 1


def test_run_refold_protenix_passes_constraint_weight_path_into_evaluator_conf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from omegaconf import OmegaConf

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    src_run_dir.mkdir(parents=True, exist_ok=True)
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "weight_path:",
                "  protenix_constraint: /tmp/protenix_constraint.pt",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5,6",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["protenix"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_kwargs: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"AB": "input"},
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )

    captured: dict[str, object] = {}

    class _FakeProtenixEvaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return None

    def _construct_evaluators(cfg, _out_root, oligo_names):
        assert oligo_names == ["AB"]
        captured["constraint_ckpt"] = OmegaConf.select(cfg, "weight_path.protenix_constraint")
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_protenix"] == 1
    assert captured["constraint_ckpt"] == "/tmp/protenix_constraint.pt"


def test_run_refold_accepts_workers_for_parallel_group_queue(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(tmp_path / "source_a"),
                        "src_config": str(tmp_path / "source_a" / "config.yaml"),
                        "seqs": {"A": "AAAA", "B": "BBBB"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
                json.dumps(
                    {
                        "task": "task_b",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(tmp_path / "source_b"),
                        "src_config": str(tmp_path / "source_b" / "config.yaml"),
                        "seqs": {"A": "CCCC", "B": "DDDD"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        workers=2,
        dry_run=True,
    )

    assert summary["failed"] == 0
    assert summary["processed"] == 2
    assert summary["workers_requested"] == 2
    assert summary["workers"] == 1


def test_run_refold_uses_one_fixed_job_per_backend_slot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc import refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    manifest.write_text(
        json.dumps(item) + "\n" + json.dumps(item) + "\n",
        encoding="utf-8",
    )
    _install_fake_af3(monkeypatch)
    original = refold.run_fixed_task_list
    calls = []

    def observe(tasks, worker, **kwargs):
        tasks = tuple(tasks)
        calls.append((tasks, kwargs))
        return original(tasks, worker, **kwargs)

    monkeypatch.setattr(refold, "run_fixed_task_list", observe)
    summary = refold.run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert summary["ran_af3"] == 1
    assert summary["receipt_cache_hits_af3"] == 1
    assert len(calls) == 1
    tasks, kwargs = calls[0]
    assert len(tasks) == 2
    assert kwargs["max_in_flight"] == 1
    assert kwargs["worker_execution"] == "synchronous"
    assert [task.payload.backend for task in tasks] == ["af3", "af3"]
    assert len({task.job_id for task in tasks}) == 2


def test_parallel_refold_uses_one_fixed_job_per_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.design.fixed_list import FixedTaskListRun
    from astrolabe.posthoc import refold, refold_plan

    shards = [
        [
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "A",
                "seqs": {"A": "AAAA"},
                "types": {"A": "protein"},
                "_refold_dst_run_name": "task_a",
            }
        ],
        [
            {
                "task": "task_b",
                "run": "001_run",
                "step": 0,
                "oligo": "A",
                "seqs": {"A": "BBBB"},
                "types": {"A": "protein"},
                "_refold_dst_run_name": "task_b",
            }
        ],
    ]
    monkeypatch.setattr(
        refold_plan,
        "assign_refold_plan_to_worker_shards",
        lambda **_kwargs: (shards, 2),
    )
    observed = {}

    def fake_fixed(tasks, worker, **kwargs):
        observed["tasks"] = tuple(tasks)
        observed["worker"] = worker
        observed["max_in_flight"] = kwargs["max_in_flight"]
        return FixedTaskListRun(collections=())

    monkeypatch.setattr(refold, "run_fixed_task_list", fake_fixed)
    summary = refold._run_refold_parallel(
        plan=SimpleNamespace(groups=(object(), object())),
        manifest_jsonl=tmp_path / "manifest.jsonl",
        out_root=tmp_path / "out",
        evaluators=["af3"],
        binder_chain="B",
        target_chain="A",
        relpath_base=None,
        path_map=(),
        keep_protenix_structures=False,
        config_yaml=None,
        workers=2,
    )

    assert summary["workers"] == 2
    assert observed["max_in_flight"] == 2
    assert [task.job_id for task in observed["tasks"]] == [
        "shard_0001",
        "shard_0002",
    ]
    assert [task.payload.index for task in observed["tasks"]] == [1, 2]
    assert len({id(task.payload.command) for task in observed["tasks"]}) == 2


def test_parallel_refold_collects_popen_waits_concurrently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc import refold, refold_plan

    shards = [
        [
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "A",
                "seqs": {"A": "AAAA"},
                "types": {"A": "protein"},
                "_refold_dst_run_name": "task_a",
            }
        ],
        [
            {
                "task": "task_b",
                "run": "001_run",
                "step": 0,
                "oligo": "A",
                "seqs": {"A": "BBBB"},
                "types": {"A": "protein"},
                "_refold_dst_run_name": "task_b",
            }
        ],
    ]
    monkeypatch.setattr(
        refold_plan,
        "assign_refold_plan_to_worker_shards",
        lambda **_kwargs: (shards, 2),
    )
    entered = threading.Barrier(2)
    waited: list[str] = []
    wait_lock = threading.Lock()

    class FakeProcess:
        def __init__(self, command: list[str], **_kwargs) -> None:
            Path(command[-1]).write_text("{}\n", encoding="utf-8")

        def wait(self) -> int:
            with wait_lock:
                waited.append(threading.current_thread().name)
            entered.wait(timeout=5)
            return 0

    monkeypatch.setattr(refold.subprocess, "Popen", FakeProcess)
    summary = refold._run_refold_parallel(
        plan=SimpleNamespace(groups=(object(), object())),
        manifest_jsonl=tmp_path / "manifest.jsonl",
        out_root=tmp_path / "out",
        evaluators=["af3"],
        binder_chain="B",
        target_chain="A",
        relpath_base=None,
        path_map=(),
        keep_protenix_structures=False,
        config_yaml=None,
        workers=2,
    )

    assert len(waited) == 2
    assert len(set(waited)) == 2
    assert summary["workers"] == 2
    assert summary["groups"] == 2


def test_run_refold_separates_outputs_for_distinct_source_runs_with_same_task_and_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source" / "seed_1" / "task_a" / "001_run"
    source_b = tmp_path / "source" / "seed_2" / "task_a" / "001_run"
    for source in (source_a, source_b):
        source.mkdir(parents=True, exist_ok=True)
        (source / "config.yaml").write_text(
            "\n".join(
                [
                    "seed: 1",
                    "evaluator:",
                    "  evaluator_terms:",
                    "    AF3Alias:",
                    "      _type: alphafold3",
                    "  alphafold3:",
                    "    model_dir: /tmp/af3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(source_a),
                        "src_config": str(source_a / "config.yaml"),
                        "seqs": {"A": "AAAA", "B": "BBBB"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(source_b),
                        "src_config": str(source_b / "config.yaml"),
                        "seqs": {"A": "CCCC", "B": "DDDD"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    scores_paths = sorted((tmp_path / "refold_runs" / "task_a").rglob("scores.csv"))
    assert summary["failed"] == 0
    assert summary["processed"] == 2
    assert len(scores_paths) == 2

    observed_sources = set()
    for scores_path in scores_paths:
        with scores_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        observed_sources.add(rows[0]["src_run_dir"])

    assert observed_sources == {str(source_a), str(source_b)}


def test_run_refold_parallel_separates_outputs_for_distinct_source_runs_with_same_task_and_run(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    model_dir = tmp_path / "af3-model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"af3")
    source_a = tmp_path / "source" / "seed_1" / "task_a" / "001_run"
    source_b = tmp_path / "source" / "seed_2" / "task_a" / "001_run"
    for source in (source_a, source_b):
        source.mkdir(parents=True, exist_ok=True)
        af3_dir = source / "raw_model_results" / "alphafold3" / "step_000_AB"
        af3_dir.mkdir(parents=True, exist_ok=True)
        (af3_dir / "step_000_AB_summary_confidences.json").write_text("{}\n", encoding="utf-8")
        (source / "structures").mkdir(parents=True, exist_ok=True)
        (source / "structures" / "step_000_AB_model.cif").write_text("data\n", encoding="utf-8")
        (source / "config.yaml").write_text(
            "\n".join(
                [
                    "seed: 1",
                    "evaluator:",
                    "  evaluator_terms:",
                    "    AF3Alias:",
                    "      _type: alphafold3",
                    "  alphafold3:",
                    f"    model_dir: {model_dir}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    items = [
        {
            "task": "task_a",
            "run": "001_run",
            "step": 0,
            "oligo": "AB",
            "src_run_dir": str(source_a),
            "src_config": str(source_a / "config.yaml"),
            "seqs": {"A": "AAAA", "B": "BBBB"},
            "types": {"A": "protein", "B": "protein"},
            "homologies": {"A": None, "B": None},
            "needed_evaluators": ["af3"],
        },
        {
            "task": "task_a",
            "run": "001_run",
            "step": 0,
            "oligo": "AB",
            "src_run_dir": str(source_b),
            "src_config": str(source_b / "config.yaml"),
            "seqs": {"A": "CCCC", "B": "DDDD"},
            "types": {"A": "protein", "B": "protein"},
            "homologies": {"A": None, "B": None},
            "needed_evaluators": ["af3"],
        },
    ]
    for item in items:
        _write_source_receipt(item, real_assets=True)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(item) for item in items
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        workers=2,
    )

    scores_paths = sorted((tmp_path / "refold_runs" / "task_a").rglob("scores.csv"))
    run_dir_names = {path.parent.name for path in scores_paths}
    assert summary["failed"] == 0
    assert summary["reused_af3"] == 2
    assert summary["workers"] == 2
    assert len(scores_paths) == 2
    assert len(run_dir_names) == 2
    assert any("seed_1" in name for name in run_dir_names)
    assert any("seed_2" in name for name in run_dir_names)

    observed_sources = set()
    for scores_path in scores_paths:
        with scores_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        observed_sources.add(rows[0]["src_run_dir"])

    assert observed_sources == {str(source_a), str(source_b)}

    scores_before_cache = {
        path: path.read_bytes() for path in scores_paths
    }
    receipt_paths = sorted(
        (tmp_path / "refold_runs" / "task_a").rglob(
            ".framework/refold/af3/step_000_AB/receipt.json"
        )
    )
    receipts_before_cache = {
        path: path.read_bytes() for path in receipt_paths
    }
    assert len(receipts_before_cache) == 2
    assert all(
        json.loads(payload)["status"] == "completed"
        for payload in receipts_before_cache.values()
    )

    cached = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        workers=2,
    )

    assert cached["failed"] == 0
    assert cached["processed"] == 0
    assert cached["skipped"] == 2
    assert cached["ran_af3"] == 0
    assert cached["reused_af3"] == 0
    assert cached["receipt_cache_hits_af3"] == 2
    assert cached["evaluator_cache_misses"] == 0
    assert {
        path: path.read_bytes() for path in scores_paths
    } == scores_before_cache
    assert {
        path: path.read_bytes() for path in receipt_paths
    } == receipts_before_cache
    assert len(
        list(
            (tmp_path / "refold_runs" / "_parallel_workers").glob(
                "run_*/generation.json"
            )
        )
    ) == 2

    items[0]["seqs"]["A"] = "GGGG"
    manifest_path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n",
        encoding="utf-8",
    )
    conflicted = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        workers=2,
    )

    assert conflicted["failed"] == 1
    assert conflicted["ran_af3"] == 0
    assert conflicted["reused_af3"] == 0
    assert conflicted["receipt_cache_hits_af3"] == 1
    failure = json.loads(
        Path(conflicted["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert failure["reason"] == "refold_receipt_conflict"
    assert "scientific identity" in failure["error"]

    items[0]["seqs"]["A"] = "BBBB"
    manifest_path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n",
        encoding="utf-8",
    )
    seed_1_run_name = next(
        path.parent.name for path in scores_paths if "seed_1" in path.parent.name
    )
    issued_path = next(
        path for path in receipt_paths if seed_1_run_name in path.parts
    )
    issued_receipt = json.loads(issued_path.read_text(encoding="utf-8"))
    issued_receipt["status"] = "issued"
    issued_receipt.pop("outputs")
    issued_receipt.pop("completed_unix_s")
    issued_path.write_text(json.dumps(issued_receipt), encoding="utf-8")

    stopped = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        workers=2,
    )

    assert stopped["failed"] == 1
    assert stopped["ran_af3"] == 0
    assert stopped["receipt_cache_hits_af3"] == 1
    assert stopped["evaluator_cache_misses"] == 0
    stopped_failure = json.loads(
        Path(stopped["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert "issued without a terminal receipt" in stopped_failure["error"]

    issued_path.write_bytes(receipts_before_cache[issued_path])
    legacy = tmp_path / "refold_runs" / "_parallel_workers" / "worker_1.out"
    legacy.mkdir()
    with pytest.raises(
        refold_receipts.RefoldReceiptConflict,
        match="legacy temporary-output layout",
    ):
        run_refold(
            manifest_jsonl=manifest_path,
            out_root=tmp_path / "refold_runs",
            evaluators=["af3"],
            workers=2,
        )


def _write_refold_af3_config(path: Path, *, model_dir: str = "/tmp/af3") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                f"    model_dir: {model_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _refold_manifest_item(
    *,
    source: Path,
    task: str,
    sequence_a: str = "AAAA",
    needed_evaluators: list[str] | None = None,
) -> dict[str, object]:
    return {
        "task": task,
        "run": "001_run",
        "step": 0,
        "oligo": "AB",
        "src_run_dir": str(source),
        "src_config": str(source / "config.yaml"),
        "seqs": {"A": sequence_a, "B": "BBBB"},
        "types": {"A": "protein", "B": "protein"},
        "homologies": {"A": None, "B": None},
        "needed_evaluators": needed_evaluators or ["af3"],
    }


def _write_refold_protenix_config(path: Path, *, checkpoint: str = "/tmp/protenix.pt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                f"    load_checkpoint_path: {checkpoint}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_refold_reuses_evaluator_init_across_output_groups_with_same_resource(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    _write_refold_af3_config(source_a / "config.yaml")
    _write_refold_af3_config(source_b / "config.yaml")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(_refold_manifest_item(source=source_a, task="task_a")),
                json.dumps(_refold_manifest_item(source=source_b, task="task_b", sequence_a="CCCC")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}
    observed_output_dirs: list[str] = []

    class _FakeAF3Evaluator:
        output_dir = None

        def evaluate(self, _inputs, runtime_config=None):
            observed_output_dirs.append(str(self.output_dir))
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 2
    assert summary["groups"] == 2
    assert summary["resource_groups"] == 1
    assert summary["evaluator_cache_misses"] == 1
    assert summary["evaluator_cache_hits"] == 1
    assert calls["construct"] == 1
    assert len(set(observed_output_dirs)) == 2
    assert all(path.endswith("raw_model_results/alphafold3") for path in observed_output_dirs)


def test_run_refold_retargets_cached_protenix_runner_across_output_groups(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    _write_refold_protenix_config(source_a / "config.yaml")
    _write_refold_protenix_config(source_b / "config.yaml")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    _refold_manifest_item(
                        source=source_a,
                        task="task_a",
                        needed_evaluators=["protenix"],
                    )
                ),
                json.dumps(
                    _refold_manifest_item(
                        source=source_b,
                        task="task_b",
                        sequence_a="CCCC",
                        needed_evaluators=["protenix"],
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr("astrolabe.posthoc.refold._merge_backend_metrics_into_rows", lambda **_kwargs: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}
    observed_dump_dirs: list[str] = []
    observed_dumper_dirs: list[str] = []

    class _FakeProtenixEvaluator:
        def __init__(self):
            self.model_runner = SimpleNamespace(
                configs=SimpleNamespace(dump_dir=None),
                dumper=SimpleNamespace(base_dir=None),
            )

        def evaluate(self, _inputs, runtime_config=None):
            observed_dump_dirs.append(str(self.model_runner.configs.dump_dir))
            observed_dumper_dirs.append(str(self.model_runner.dumper.base_dir))
            return None

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_protenix"] == 2
    assert summary["resource_groups"] == 1
    assert summary["evaluator_cache_misses"] == 1
    assert summary["evaluator_cache_hits"] == 1
    assert calls["construct"] == 1
    assert observed_dump_dirs == observed_dumper_dirs
    assert len(set(observed_dump_dirs)) == 2
    assert all(path.endswith("raw_model_results/protenix") for path in observed_dump_dirs)


def test_run_refold_keeps_evaluator_init_separate_for_different_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    _write_refold_af3_config(source_a / "config.yaml", model_dir="/tmp/af3_a")
    _write_refold_af3_config(source_b / "config.yaml", model_dir="/tmp/af3_b")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(_refold_manifest_item(source=source_a, task="task_a")),
                json.dumps(_refold_manifest_item(source=source_b, task="task_b", sequence_a="CCCC")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}

    class _FakeAF3Evaluator:
        output_dir = None

        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 2
    assert summary["resource_groups"] == 2
    assert summary["evaluator_cache_misses"] == 2
    assert summary["evaluator_cache_hits"] == 0
    assert calls["construct"] == 2


def test_refold_parallel_shards_keep_same_resource_groups_together(tmp_path: Path) -> None:
    from astrolabe.posthoc import refold_plan

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    source_c = tmp_path / "source_c" / "task_c" / "001_run"
    _write_refold_af3_config(source_a / "config.yaml", model_dir="/tmp/af3_shared")
    _write_refold_af3_config(source_b / "config.yaml", model_dir="/tmp/af3_shared")
    _write_refold_af3_config(source_c / "config.yaml", model_dir="/tmp/af3_other")

    item_a = _refold_manifest_item(source=source_a, task="task_a")
    item_b = _refold_manifest_item(source=source_b, task="task_b")
    item_c = _refold_manifest_item(source=source_c, task="task_c")
    plan = refold_plan.build_refold_plan(
        items=[item_a, item_b, item_c],
        requested_backends={"af3"},
        config_yaml=None,
        load_runtime_signature=refold_plan_runtime_signature,
    )

    shards, resource_groups = refold_plan.assign_refold_plan_to_worker_shards(
        plan=plan,
        worker_count=2,
    )

    locations = {
        str(item["task"]): shard_idx
        for shard_idx, shard in enumerate(shards)
        for item in shard
    }
    assert resource_groups == 2
    assert locations["task_a"] == locations["task_b"]
    assert locations["task_c"] != locations["task_a"]


def refold_plan_runtime_signature(path: Path, backend: str) -> dict[str, str]:
    if backend == "af3":
        text = Path(path).read_text(encoding="utf-8")
        model_dir = next(
            (line.split(":", 1)[1].strip() for line in text.splitlines() if "model_dir:" in line),
            "",
        )
        return {"backend": backend, "model_dir": model_dir}
    return {"backend": backend, "path": str(path)}


def test_run_refold_dry_run_accepts_manifest_without_src_config(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "dummy_task",
                "run": "001_run",
                "step": 1,
                "oligo": "AB",
                "seqs": {"A": "ACDE", "B": "FGHI"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "src_run_dir": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        dry_run=True,
        workers=3,
    )

    assert summary["failed"] == 0
    assert summary["processed"] == 1
    assert summary["workers_requested"] == 3
    assert summary["workers"] == 1
    assert (tmp_path / "refold_runs" / "dummy_task" / "001_run" / "scores.csv").exists()


def test_run_refold_dry_run_does_not_consume_source_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    source = Path(item["src_run_dir"])
    marker = (
        source
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text('{"source": "completed"}\n', encoding="utf-8")
    structure = source / "structures" / "step_000_A_model.cif"
    structure.parent.mkdir(parents=True)
    structure.write_text("data_source\n#\n", encoding="utf-8")
    _write_source_receipt(item)
    calls = _install_fake_af3(monkeypatch)

    summary = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
        dry_run=True,
    )

    target = out_root / "task_a" / "001_run"
    assert summary["failed"] == 0
    assert summary["processed"] == 1
    assert summary["reused_af3"] == 0
    assert summary["ran_af3"] == 0
    assert calls == {"construct": 0, "evaluate": 0}
    assert (target / "scores.csv").exists()
    assert not (target / "raw_model_results" / "alphafold3").exists()
    assert not (target / "structures" / "step_000_A_model.cif").exists()
    assert not (
        target
        / ".framework"
        / "refold"
        / "af3"
        / "step_000_A"
        / "receipt.json"
    ).exists()


def test_refold_backend_selection_rejects_unknown_backend() -> None:
    from astrolabe.posthoc import refold_backends

    with pytest.raises(ValueError, match="Unsupported refold backend"):
        refold_backends.refold_backend_names(["af3", "esmfold"])


def test_run_refold_rejects_unknown_backend_before_creating_output(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "dummy_task",
                "run": "001_run",
                "step": 1,
                "oligo": "AB",
                "seqs": {"A": "ACDE", "B": "FGHI"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "src_run_dir": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "refold_runs"

    with pytest.raises(ValueError, match="Unsupported refold backend"):
        run_refold(
            manifest_jsonl=manifest_path,
            out_root=out_root,
            evaluators=["esmfold"],
            dry_run=True,
        )

    assert not out_root.exists()


def test_export_refold_manifest_includes_passing_rows_when_passed_only_disabled(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_refold_record(
        db_path=db_path,
        run_dir=run_dir,
        metrics={
            "stage": "default",
            "weighted_score": 1.0,
            "pA_sequence": "AAAA",
            "pB_sequence": "BBBB",
            "protenix_iptm": 0.91,
        },
    )
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
        passed_only=False,
    )

    assert summary["records_passed"] == 1
    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["src_run_dir"] == str(run_dir.resolve())
    assert payload["needed_evaluators"] == []


def test_completed_receipt_reuses_only_identical_scientific_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    calls = _install_fake_af3(monkeypatch)
    first = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )
    assert first["ran_af3"] == 1
    assert calls == {"construct": 1, "evaluate": 1}
    scores_path = out_root / "task_a" / "001_run" / "scores.csv"
    scores_before = scores_path.read_bytes()

    second = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )
    assert second["failed"] == 0
    assert second["skipped"] == 1
    assert second["receipt_cache_hits_af3"] == 1
    assert calls == {"construct": 1, "evaluate": 1}
    assert scores_path.read_bytes() == scores_before

    run_dir = out_root / "task_a" / "001_run"
    marker = (
        run_dir
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    before = marker.read_bytes()
    item["seqs"]["A"] = "CCCC"
    manifest.write_text(json.dumps(item) + "\n", encoding="utf-8")
    changed = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert changed["failed"] == 1
    assert changed["ran_af3"] == 0
    assert calls == {"construct": 1, "evaluate": 1}
    assert marker.read_bytes() == before
    failure = json.loads(
        Path(changed["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert failure["reason"] == "refold_receipt_conflict"
    assert "scientific identity" in failure["error"]


def test_same_path_config_change_rejects_stale_target_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    calls = _install_fake_af3(monkeypatch)
    first = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )
    assert first["failed"] == 0

    config_path = Path(item["src_config"])
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# changed in place\n",
        encoding="utf-8",
    )
    changed = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert changed["failed"] == 1
    assert changed["ran_af3"] == 0
    assert calls == {"construct": 1, "evaluate": 1}


def test_same_path_weight_change_changes_selected_asset_identity(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "af3-model"
    model_dir.mkdir()
    weights = model_dir / "af3.bin"
    weights.write_bytes(b"weights-v1")
    config = {"model_dir": str(model_dir)}
    first = _REAL_SELECTED_ASSET_RECORD(
        backend="af3",
        selected_config=config,
    )
    weights.write_bytes(b"weights-v2")
    second = _REAL_SELECTED_ASSET_RECORD(
        backend="af3",
        selected_config=config,
    )

    assert first["complete"] is True
    assert second["complete"] is True
    assert first["files"][0]["path"] == second["files"][0]["path"]
    assert first["files"][0]["sha256"] != second["files"][0]["sha256"]


def test_same_path_weight_change_rejects_target_receipt_before_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    model_dir = tmp_path / "af3-model"
    model_dir.mkdir()
    weights = model_dir / "af3.bin"
    weights.write_bytes(b"weights-v1")
    _write_refold_af3_config(
        Path(item["src_config"]),
        model_dir=str(model_dir),
    )
    monkeypatch.setattr(
        refold_receipts,
        "selected_asset_record",
        _REAL_SELECTED_ASSET_RECORD,
    )
    calls = _install_fake_af3(monkeypatch)
    first = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )
    assert first["failed"] == 0

    weights.write_bytes(b"weights-v2")
    changed = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert changed["failed"] == 1
    assert changed["ran_af3"] == 0
    assert calls == {"construct": 1, "evaluate": 1}


def test_af3_numeric_source_hashes_runtime_package_origin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "site-packages" / "alphafold3"
    package.mkdir(parents=True)
    source = package / "model.py"
    source.write_bytes(b"package-source-v1\n")
    for relative in (
        "astrolabe/posthoc/refold.py",
        "astrolabe/posthoc/refold_backends.py",
        "astrolabe/posthoc/refold_plan.py",
        "astrolabe/posthoc/refold_receipts.py",
        "astrolabe/evaluators/base.py",
        "astrolabe/evaluators/utils.py",
        "astrolabe/evaluators/alphafold3.py",
        "astrolabe/evaluators/wrappers/alphafold3.py",
        "astrolabe/design/fixed_list.py",
        "astrolabe/design/collection.py",
        "astrolabe/design/collection_search.py",
        "astrolabe/design/coordinator.py",
        "astrolabe/design/composition.py",
        "run.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(
        refold_receipts.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(package)])
            if name == "alphafold3"
            else None
        ),
    )

    first = _REAL_NUMERIC_SOURCE_RECORD("af3", repo_root=tmp_path)
    source.write_bytes(b"package-source-v2\n")
    second = _REAL_NUMERIC_SOURCE_RECORD("af3", repo_root=tmp_path)

    assert first["origin"] == {
        "kind": "python-package",
        "module": "alphafold3",
        "directories": [str(package.resolve())],
    }
    assert first["schema"] == (
        "astrolabe.posthoc-refold.numeric-source/v2"
    )
    assert first["scope"]["own_package"] == (
        "all astrolabe/**/*.py plus run.py"
    )
    source_paths = {item["path"] for item in first["files"]}
    for relative in (
        "astrolabe/design/fixed_list.py",
        "astrolabe/design/collection.py",
        "astrolabe/design/collection_search.py",
        "astrolabe/design/coordinator.py",
        "astrolabe/design/composition.py",
        "run.py",
    ):
        assert str((tmp_path / relative).resolve()) in source_paths
    first_package = next(
        item for item in first["files"] if item["path"] == str(source.resolve())
    )
    second_package = next(
        item for item in second["files"] if item["path"] == str(source.resolve())
    )
    assert first_package["sha256"] != second_package["sha256"]
    assert all(
        "/vendor/alphafold3" not in item["path"]
        for item in first["files"]
    )


def test_target_marker_without_receipt_is_an_explicit_conflict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    run_dir = out_root / "task_a" / "001_run"
    marker = (
        run_dir
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    calls = _install_fake_af3(monkeypatch)

    summary = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert summary["failed"] == 1
    assert summary["ran_af3"] == 0
    assert calls == {"construct": 0, "evaluate": 0}
    failure = json.loads(
        Path(summary["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert "no committed receipt" in failure["error"]


def test_legacy_source_without_receipt_is_not_reused_but_target_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    source = Path(item["src_run_dir"])
    marker = (
        source
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    structure = source / "structures" / "step_000_A_model.cif"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    structure.parent.mkdir(parents=True)
    structure.write_text("data_source\n#\n", encoding="utf-8")
    calls = _install_fake_af3(monkeypatch)

    summary = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["reused_af3"] == 0
    assert summary["ran_af3"] == 1
    assert calls == {"construct": 1, "evaluate": 1}


def test_receipt_is_issued_before_model_callback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, _item = _posthoc_af3_case(tmp_path)
    _install_fake_af3(monkeypatch)
    observed = []

    class ObservingAF3:
        def evaluate(self, _inputs, runtime_config=None):
            receipt = (
                out_root
                / "task_a"
                / "001_run"
                / ".framework"
                / "refold"
                / "af3"
                / "step_000_A"
                / "receipt.json"
            )
            observed.append(json.loads(receipt.read_text(encoding="utf-8")))
            return SimpleNamespace(
                iptm={"A": [0.8]},
                ptm={"A": [0.7]},
                plddt={"A": [0.9]},
            )

    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_evaluators",
        lambda *_args, **_kwargs: {"Alphafold3": ObservingAF3()},
    )
    summary = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert len(observed) == 1
    assert observed[0]["status"] == "issued"


def test_issued_receipt_with_marker_is_unknown_and_never_replayed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    calls = _install_fake_af3(monkeypatch)
    first = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )
    assert first["failed"] == 0

    run_dir = out_root / "task_a" / "001_run"
    receipt_path = (
        run_dir
        / ".framework"
        / "refold"
        / "af3"
        / "step_000_A"
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "issued"
    receipt.pop("outputs")
    receipt.pop("completed_unix_s")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    stopped = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )
    assert stopped["failed"] == 1
    assert stopped["ran_af3"] == 0
    assert calls == {"construct": 1, "evaluate": 1}
    failure = json.loads(
        Path(stopped["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert "issued without a terminal receipt" in failure["error"]


def test_marker_written_during_failed_model_call_stays_issued_and_never_replays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, _item = _posthoc_af3_case(tmp_path)
    calls = {"construct": 0, "evaluate": 0}

    class FailingAF3:
        def evaluate(self, _inputs, runtime_config=None):
            calls["evaluate"] += 1
            run_dir = out_root / "task_a" / "001_run"
            marker = (
                run_dir
                / "raw_model_results"
                / "alphafold3"
                / "step_000_A"
                / "step_000_A_summary_confidences.json"
            )
            structure = run_dir / "structures" / "step_000_A_model.cif"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text('{"partial": true}\n', encoding="utf-8")
            structure.parent.mkdir(parents=True, exist_ok=True)
            structure.write_text("data_partial\n#\n", encoding="utf-8")
            raise RuntimeError("model call failed after writing marker")

    def construct(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["A"]
        return {"Alphafold3": FailingAF3()}

    monkeypatch.setattr(
        "astrolabe.posthoc.refold._init_refold_global_config",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._build_protomers",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"A": "input"},
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_evaluators",
        construct,
    )

    with pytest.raises(RuntimeError, match="failed after writing marker"):
        run_refold(
            manifest_jsonl=manifest,
            out_root=out_root,
            evaluators=["af3"],
        )

    receipt_path = (
        out_root
        / "task_a"
        / "001_run"
        / ".framework"
        / "refold"
        / "af3"
        / "step_000_A"
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "issued"
    assert "outputs" not in receipt

    stopped = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert stopped["failed"] == 1
    assert stopped["ran_af3"] == 0
    assert calls == {"construct": 1, "evaluate": 1}
    failure = json.loads(
        Path(stopped["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert "issued without a terminal receipt" in failure["error"]


def test_duplicate_manifest_slot_issues_only_one_model_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, item = _posthoc_af3_case(tmp_path)
    manifest.write_text(
        json.dumps(item) + "\n" + json.dumps(item) + "\n",
        encoding="utf-8",
    )
    calls = _install_fake_af3(monkeypatch)

    summary = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["processed"] == 1
    assert summary["skipped"] == 1
    assert summary["ran_af3"] == 1
    assert summary["receipt_cache_hits_af3"] == 1
    assert calls == {"construct": 1, "evaluate": 1}


def test_completed_receipt_rejects_changed_output_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest, out_root, _item = _posthoc_af3_case(tmp_path)
    calls = _install_fake_af3(monkeypatch)
    first = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )
    assert first["failed"] == 0

    marker = (
        out_root
        / "task_a"
        / "001_run"
        / "raw_model_results"
        / "alphafold3"
        / "step_000_A"
        / "step_000_A_summary_confidences.json"
    )
    marker.write_text('{"changed": true}\n', encoding="utf-8")
    changed = run_refold(
        manifest_jsonl=manifest,
        out_root=out_root,
        evaluators=["af3"],
    )

    assert changed["failed"] == 1
    assert changed["ran_af3"] == 0
    assert calls == {"construct": 1, "evaluate": 1}
    failure = json.loads(
        Path(changed["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert "outputs changed" in failure["error"]


def test_receipt_slot_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    slot = refold_receipts.RefoldReceiptSlot(
        tmp_path / "run",
        "af3",
        0,
        "A",
    )
    errors: list[BaseException] = []

    with slot.lock():
        def contend() -> None:
            try:
                with slot.lock():
                    pass
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=contend)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()

    assert len(errors) == 1
    assert isinstance(errors[0], refold_receipts.RefoldReceiptConflict)
    assert "already active" in str(errors[0])
    assert slot.lock_path.read_bytes() == b""


def test_receipt_slot_fcntl_lock_rejects_another_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    run_dir = str(tmp_path / "run")

    def hold_slot() -> None:
        slot = refold_receipts.RefoldReceiptSlot(
            Path(run_dir),
            "af3",
            0,
            "A",
        )
        with slot.lock():
            ready.set()
            release.wait(10)

    process = context.Process(target=hold_slot)
    process.start()
    try:
        assert ready.wait(5)
        slot = refold_receipts.RefoldReceiptSlot(
            Path(run_dir),
            "af3",
            0,
            "A",
        )
        with pytest.raises(
            refold_receipts.RefoldReceiptConflict,
            match="slot lock is held",
        ):
            with slot.lock():
                pass
    finally:
        release.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0
