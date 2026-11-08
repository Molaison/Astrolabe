from __future__ import annotations

import math
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design import native_flow
from astrolabe.design.legacy import standard_scores_to_evidence, translate_legacy_state
from astrolabe.design.store import ArrayCodec, DataStore


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CAPTURE = (
    _REPO_ROOT
    / "outputs/unified-gpu-acceptance/af3_default_api/"
    "resumes/default_api_001/capture.pickle"
)


@pytest.fixture(autouse=True)
def clean_legacy_globals():
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


def _read_capture_records(path: Path) -> list[dict]:
    records = []
    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return records
            assert len(header) == 8
            size = int.from_bytes(header, "big")
            payload = handle.read(size)
            assert len(payload) == size
            records.append(pickle.loads(payload))


def _captured_af3_scores() -> tuple[object, Path]:
    capture = Path(os.environ.get("ASTROLABE_AF3_CAPTURE", _DEFAULT_CAPTURE))
    if not capture.is_file():
        pytest.skip(f"real AF3 capture is unavailable: {capture}")
    records = _read_capture_records(capture)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "ok"
    assert record["evaluator_alias"] == "Alphafold3"
    scores = record["scores"]
    structure = capture.parent / "evidence/structures/evaluation_000__A.cif"
    assert structure.is_file()
    # The deployed evidence mount used a runtime-only path. Rebind that path to
    # the immutable captured bytes without changing any scientific score array.
    scores.complex_paths = {"A": structure}
    return scores, capture


def test_real_af3_28424_capture_replays_nonfinite_scores_on_cpu():
    scores, capture = _captured_af3_scores()
    protomer = Protomer("A", "AC", "p")
    global_config.protomers["A"] = protomer
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)

    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
    )

    iptm = next(
        item
        for item in evidence.measurements
        if ".iptm.A.prediction_0" in item.quantity
    )
    iptm_values = data.get(iptm.values, ArrayCodec())
    iptm_mask = data.get(iptm.valid_mask, ArrayCodec())
    assert iptm.missing == "partial"
    assert iptm_values.shape == ()
    assert np.isnan(iptm_values.item())
    assert iptm_mask.shape == ()
    assert bool(iptm_mask.item()) is False

    pae = next(
        item
        for item in evidence.measurements
        if ".pae.A.prediction_0" in item.quantity
    )
    pae_values = data.get(pae.values, ArrayCodec())
    assert pae.missing == "partial"
    assert np.isnan(pae_values).all()
    assert not data.get(pae.valid_mask, ArrayCodec()).any()

    plddt = next(
        item
        for item in evidence.measurements
        if ".plddt.A.prediction_0" in item.quantity
    )
    assert plddt.missing == "present"
    assert plddt.valid_mask is None
    assert capture.name == "capture.pickle"
    assert np.isnan(scores.iptm["A"][0])
    assert np.isnan(scores.chain_pair_scores["A"]["chain_iptm"][0][0])


@pytest.mark.parametrize("objective", [math.nan, math.inf, -math.inf])
def test_native_projection_rejects_nonfinite_weighted_objective(objective):
    scores, _capture = _captured_af3_scores()
    protomer = Protomer("A", "AC", "p")
    global_config.protomers["A"] = protomer
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    resources = SimpleNamespace(
        retain=lambda *_args, **_kwargs: None,
    )
    decisions = SimpleNamespace(scores={0: objective}, last_accepted=0)
    history = SimpleNamespace(revision=3)

    with pytest.raises(ValueError, match="weighted objective must be finite"):
        native_flow._project(
            scores,
            translation,
            data,
            resources,
            decisions,
            SimpleNamespace(current_stage="default", loss_computer=None),
            0,
            history,
        )
