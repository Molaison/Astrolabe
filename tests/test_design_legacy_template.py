from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.legacy import (
    LEGACY_REFERENCE_CODECS,
    LegacyContext,
    LegacyTemplateCodec,
    UnsupportedLegacyState,
    decode_legacy_templates,
    translate_legacy_state,
)
from astrolabe.design.store import DataStore, JSONCodec
from astrolabe.design.types import (
    Assignment,
    Blob,
    Conditioning,
    Dependency,
    Key,
    Polymer,
    key_for,
    manifest,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_REFERENCE = "$legacy_template/v1"


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


def _add_archived_protomer() -> Protomer:
    cfg = OmegaConf.load(ROOT / "config" / "paper" / "bivalent_rna.yaml")
    sequence = str(OmegaConf.to_container(cfg.seqs, resolve=True)["B"]).removeprefix(
        "+"
    )
    homology = OmegaConf.to_container(cfg.homologies.B, resolve=True)
    assert isinstance(homology, dict)
    protomer = Protomer("B", sequence, "p", homology)
    global_config.protomers["B"] = protomer
    return protomer


def _topology(
    translation,
    data: DataStore,
) -> dict:
    return data.get(
        translation.system.topology,
        JSONCodec("astrolabe.legacy.topology/v1"),
    )


def _template_blob(topology: dict, protomer: str = "B") -> Blob:
    reference = topology["protomers"][protomer]["homology"]["templates"][0]
    assert set(reference) == {TEMPLATE_REFERENCE}
    payload = reference[TEMPLATE_REFERENCE]["blob"]
    return Blob(Key(payload["digest"]), payload["codec"])


def _conditioning(template: Blob) -> Conditioning:
    inputs = manifest(
        (Dependency("legacy-homology-template", template.identity),)
    )
    return Conditioning(
        identity=key_for(
            (
                "test.legacy-template-conditioning/v1",
                template,
                inputs,
            )
        ),
        owner=key_for(("test.legacy-template-conditioning-owner/v1",)),
        intent=template,
        mechanism="legacy-homology-template",
        effective_parameters=template,
        approximations=(),
        inputs=inputs,
    )


def test_archived_homology_template_survives_persist_activate_and_restore(
    tmp_path: Path,
) -> None:
    protomer = _add_archived_protomer()
    expected_template = protomer.homology["templates"][0]
    data = DataStore(tmp_path / "store")
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("B",), data=data)
    topology = _topology(translation, data)
    template = _template_blob(topology)
    reconstructed = decode_legacy_templates(
        topology["protomers"]["B"]["homology"],
        data=data,
    )["templates"][0]
    assert reconstructed == expected_template

    payload = data.get(template, LegacyTemplateCodec())
    assert payload.mmcif == expected_template.mmcif
    assert payload.query_to_template_map == expected_template.query_to_template_map
    canonical = LegacyTemplateCodec().canonical_payload(payload)
    assert canonical["mmcif"]["dtype"] == "utf-8"
    assert canonical["mmcif"]["shape"] == [
        len(payload.mmcif.encode("utf-8"))
    ]
    assert canonical["query_to_template_map"]["dtype"] == "python-int"
    assert canonical["query_to_template_map"]["shape"] == [
        len(payload.query_to_template_map),
        2,
    ]

    data.persist(
        (
            translation.system.topology,
            *(
                occurrence.index_map
                for occurrence in translation.system.occurrences
            ),
        )
    )
    cold = DataStore(tmp_path / "store")
    for codec in LEGACY_REFERENCE_CODECS:
        cold.register(codec)
    cold_topology = cold.get(
        translation.system.topology,
        JSONCodec("astrolabe.legacy.topology/v1"),
    )
    cold_template = decode_legacy_templates(
        cold_topology["protomers"]["B"]["homology"],
        data=cold,
    )["templates"][0]
    assert cold_template == expected_template

    snapshot = context.capture_state()
    original_assignment = translation.candidate.assignments[0]
    assert isinstance(original_assignment.value, Polymer)
    replacement_token = (
        "Q" if original_assignment.value.tokens[0] != "Q" else "A"
    )
    proposed = translation.system.edit(
        translation.candidate,
        (
            Assignment(
                "B_1",
                Polymer(
                    "protein",
                    (
                        replacement_token,
                        *original_assignment.value.tokens[1:],
                    ),
                    (),
                ),
            ),
        ),
        data=data,
    )
    with pytest.raises(RuntimeError, match="synthetic activation failure"):
        with context.activate(proposed, translation=translation, data=data):
            assert global_config.segments["B_1"].sequence.startswith(
                replacement_token
            )
            assert global_config.protomers["B"].homology["templates"][0] == (
                expected_template
            )
            raise RuntimeError("synthetic activation failure")

    assert global_config.segments["B_1"].sequence == "".join(
        original_assignment.value.tokens
    )
    assert global_config.protomers["B"].homology["templates"][0] == expected_template
    restored = context.capture_state()
    restored_segment = restored.segments[0].attributes
    assert restored_segment["sequence"] == "".join(
        original_assignment.value.tokens
    )
    assert restored.protomers[0].attributes["_homology"]["templates"][0] == (
        expected_template
    )


def test_template_content_changes_system_and_conditioning_identity() -> None:
    protomer = _add_archived_protomer()
    original_template = protomer.homology["templates"][0]
    first_data = DataStore()
    first = translate_legacy_state(assemblies=("B",), data=first_data)
    first_topology = _topology(first, first_data)
    first_template = _template_blob(first_topology)

    protomer._homology["templates"][0] = type(original_template)(
        mmcif=original_template.mmcif + "\n# explicit content change\n",
        query_to_template_map=original_template.query_to_template_map,
    )
    second_data = DataStore()
    second = translate_legacy_state(assemblies=("B",), data=second_data)
    second_topology = _topology(second, second_data)
    second_template = _template_blob(second_topology)

    assert first_template.identity != second_template.identity
    assert first.system.topology != second.system.topology
    assert first.system.identity != second.system.identity
    assert _conditioning(first_template).identity != _conditioning(second_template).identity


def test_template_codec_rejects_unregistered_lookalike_object() -> None:
    codec = LegacyTemplateCodec()
    with pytest.raises(UnsupportedLegacyState, match="only accepts"):
        codec.encode(
            SimpleNamespace(
                mmcif="data_template\n#\n",
                query_to_template_map={0: 0},
            )
        )
