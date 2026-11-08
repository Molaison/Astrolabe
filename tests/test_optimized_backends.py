from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from astrolabe import optimized_backends as optimized


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = optimized.load_manifest()
    source = tmp_path / "anthropic-kits"
    source.mkdir()
    (source / ".git").mkdir()
    for kit in optimized.kit_names(manifest):
        (source / kit / "opt").mkdir(parents=True)
        (source / kit / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (source / kit / "opt" / "pyproject.toml").write_text(
            "[project]\nname = 'fixture'\n", encoding="utf-8"
        )

    revision = manifest["source"]["revision"]

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return str(source)
        if args == ("rev-parse", "HEAD"):
            return revision
        raise AssertionError(f"unexpected git arguments: {args}")

    monkeypatch.setattr(optimized, "_git_output", fake_git)
    for name in ("_ACTIVATION_CACHE", "_PROCESS_ACTIVE", "_PROCESS_STATES", "_PROCESS_FAILURES"):
        monkeypatch.setattr(optimized, name, {})
    return manifest, source


def test_selection_uses_manifest_and_validated_source(fake_source) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "protenix", "exact", kit_root=source, manifest=manifest
    )

    assert selection.spec.name == "protenix_v2"
    assert selection.spec.backend == "protenix"
    assert selection.spec.software_ids == ("protenix",)
    assert selection.mode == "exact"
    assert selection.revision == manifest["source"]["revision"]
    assert selection.env == {"PROTENIX_OPT": "exact"}
    assert selection.activation_api.module == "protenix_opt"
    assert selection.model_identity.startswith("protenix-v2:2.0.0@")


def test_activation_calls_real_hook_once_and_reports_identity(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "protenix", "exact", kit_root=source, manifest=manifest
    )
    calls: list[tuple[str, bool]] = []

    def enable(mode: str, *, strict: bool) -> dict:
        calls.append((mode, strict))
        return {
            "active": True,
            "mode": mode,
            "levers_applied": ["ptx_pairformer"],
            "protenix_version": "2.0.0",
        }

    monkeypatch.setitem(
        sys.modules,
        "protenix_opt",
        SimpleNamespace(enable=enable),
    )

    first = optimized.activate_provider(selection)
    second = optimized.activate_provider(selection)

    assert calls == [("exact", True)]
    assert first is second
    assert first.active is True
    assert first.mode == "exact"
    assert first.revision == manifest["source"]["revision"]
    assert first.model_identity == selection.model_identity
    assert first.activation_api == "protenix_opt.enable"


def test_active_report_without_applied_evidence_is_unqualified(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "protenix", "exact", kit_root=source, manifest=manifest
    )
    monkeypatch.setitem(
        sys.modules,
        "protenix_opt",
        SimpleNamespace(
            enable=lambda mode, *, strict: {
                "active": True,
                "mode": mode,
            }
        ),
    )

    with pytest.raises(optimized.ActivationNotActiveError, match="unqualified"):
        optimized.activate_provider(selection)


def test_esmfold2_deferred_enable_report_is_not_active(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "esmfold2",
        "exact",
        "full_msa",
        kit_root=source,
        manifest=manifest,
    )
    monkeypatch.setitem(
        sys.modules,
        "esmfold2_opt",
        SimpleNamespace(
            enable=lambda mode, variant=None, *, strict: {
                "active": True,
                "mode": mode,
                "variant": variant,
                "applied": "deferred",
                "levers_applied": [],
            }
        ),
    )

    with pytest.raises(
        optimized.ActivationNotActiveError,
        match="deferred hook.*not yet been applied",
    ):
        optimized.activate_provider(selection)


def test_esmfold2_configured_report_is_active(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "esmfold2",
        "exact",
        "full_msa",
        kit_root=source,
        manifest=manifest,
    )
    monkeypatch.setitem(
        sys.modules,
        "esmfold2_opt",
        SimpleNamespace(
            enable=lambda mode, variant=None, *, strict: {
                "active": True,
                "mode": mode,
                "variant": variant,
                "applied": "configured",
                "levers_applied": ["ef2_trunk"],
                "partial": [],
            }
        ),
    )

    report = optimized.activate_provider(selection)

    assert report.active is True
    assert report.state == "configured"


def test_esmfold2_configured_report_requires_actual_model_levers(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "esmfold2",
        "exact",
        "full_msa",
        kit_root=source,
        manifest=manifest,
    )
    monkeypatch.setitem(
        sys.modules,
        "esmfold2_opt",
        SimpleNamespace(
            enable=lambda mode, variant=None, *, strict: {
                "active": True,
                "mode": mode,
                "variant": variant,
                "applied": "configured",
                "levers_applied": [],
                "levers_deferred": ["ef2_trunk"],
            }
        ),
    )

    with pytest.raises(
        optimized.ActivationNotActiveError,
        match="configured model levers",
    ):
        optimized.activate_provider(selection)


def test_chai_active_claim_requires_pinned_wheel_bytes(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "chai1", "exact", kit_root=source, manifest=manifest
    )
    monkeypatch.setitem(
        sys.modules,
        "chai1_opt",
        SimpleNamespace(
            enable=lambda mode, *, strict: {
                "active": True,
                "mode": mode,
                "levers_applied": ["W1", "W5"],
                "pins": {
                    "chai_lab": {
                        "pinned": False,
                        "files_differ": ["chai_lab/chai1.py"],
                    }
                },
            }
        ),
    )

    with pytest.raises(
        optimized.ActivationNotActiveError,
        match="vendored.*pinned-wheel",
    ):
        optimized.activate_provider(selection)


def test_chai_pinned_wheel_activation_is_qualified(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "chai1", "exact", kit_root=source, manifest=manifest
    )
    monkeypatch.setitem(
        sys.modules,
        "chai1_opt",
        SimpleNamespace(
            enable=lambda mode, *, strict: {
                "active": True,
                "mode": mode,
                "levers_applied": ["W1", "W5"],
                "pins": {"chai_lab": {"pinned": True}},
                "partial": False,
            }
        ),
    )

    report = optimized.activate_provider(selection)

    assert report.active is True
    assert report.state == "applied"


def test_chai_pinned_wheel_without_levers_is_unqualified(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "chai1", "exact", kit_root=source, manifest=manifest
    )
    monkeypatch.setitem(
        sys.modules,
        "chai1_opt",
        SimpleNamespace(
            enable=lambda mode, *, strict: {
                "active": True,
                "mode": mode,
                "levers_applied": [],
                "pins": {"chai_lab": {"pinned": True}},
            }
        ),
    )

    with pytest.raises(
        optimized.ActivationNotActiveError,
        match="unqualified",
    ):
        optimized.activate_provider(selection)


def test_inactive_provider_report_is_a_refusal(
    fake_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "chai1", "exact", kit_root=source, manifest=manifest
    )

    monkeypatch.setitem(
        sys.modules,
        "chai1_opt",
        SimpleNamespace(
            enable=lambda mode, *, strict: {
                "active": False,
                "mode": mode,
                "reason": "stock pin: chai_lab version mismatch",
            }
        ),
    )

    with pytest.raises(optimized.ActivationNotActiveError, match="version mismatch"):
        optimized.activate_provider(selection)


def test_source_revision_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = optimized.load_manifest()
    source = tmp_path / "drifted"
    source.mkdir()
    (source / ".git").mkdir()
    monkeypatch.setattr(
        optimized,
        "_git_output",
        lambda _root, *args: str(source)
        if args == ("rev-parse", "--show-toplevel")
        else "0" * 40,
    )

    with pytest.raises(optimized.SourceRevisionMismatchError, match="revision drift"):
        optimized.verify_source_checkout(source, manifest["source"]["revision"])


def test_source_without_own_git_marker_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty-vendor"
    source.mkdir()

    with pytest.raises(optimized.SourceUnavailableError, match="empty.*independent"):
        optimized.verify_source_checkout(source, "a" * 40)


def test_git_top_level_must_be_the_selected_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "vendor-copy"
    source.mkdir()
    (source / ".git").mkdir()
    monkeypatch.setattr(
        optimized,
        "_git_output",
        lambda _root, *args: str(tmp_path),
    )

    with pytest.raises(optimized.SourceUnavailableError, match="another repository"):
        optimized.verify_source_checkout(source, "a" * 40)


def test_boltz2_and_esmfold2_are_not_silent_replacements(fake_source) -> None:
    manifest, source = fake_source

    with pytest.raises(optimized.ProviderMismatchError, match="Boltz2"):
        optimized.build_kit_selection(
            "boltz", "exact", kit_root=source, manifest=manifest
        )
    with pytest.raises(optimized.ProviderMismatchError, match="ESMFold2"):
        optimized.build_kit_selection(
            "esmfold", "exact", kit_root=source, manifest=manifest
        )


def test_unmapped_backend_stays_native(fake_source) -> None:
    manifest, source = fake_source
    with pytest.raises(optimized.NativeBackendOnlyError, match="native backend"):
        optimized.build_kit_selection(
            "alphafold2", "exact", kit_root=source, manifest=manifest
        )


def test_unknown_mode_and_backend_fail_explicitly(fake_source) -> None:
    manifest, source = fake_source
    with pytest.raises(optimized.UnknownModeError, match="not declared"):
        optimized.build_kit_selection(
            "protenix", "turbo", kit_root=source, manifest=manifest
        )
    with pytest.raises(optimized.UnknownBackendError, match="Unknown"):
        optimized.build_kit_selection(
            "not-a-backend", "exact", kit_root=source, manifest=manifest
        )


def test_esmif_fast_is_not_a_library_activation(fake_source) -> None:
    manifest, source = fake_source
    selection = optimized.build_kit_selection(
        "esmif", "fast", kit_root=source, manifest=manifest
    )

    assert selection.env == {"ESM_IF1_OPT": "fast"}
    with pytest.raises(optimized.UnsupportedActivationError, match="probability-logits"):
        optimized.activate_provider(selection)


def test_variant_environment_is_provider_specific(fake_source) -> None:
    manifest, source = fake_source

    with pytest.raises(optimized.ProviderMismatchError, match="explicit model_variant"):
        optimized.build_kit_selection(
            "alphafold3", "exact", kit_root=source, manifest=manifest
        )
    af3 = optimized.build_kit_selection(
        "alphafold3",
        "exact",
        "p2",
        kit_root=source,
        manifest=manifest,
    )
    esmfold2 = optimized.build_kit_selection(
        "esmfold2",
        "exact",
        "full_msa",
        kit_root=source,
        manifest=manifest,
    )

    assert af3.env == {"AF3_JAX_OPT": "exact"}
    assert esmfold2.env == {
        "ESMFOLD2_OPT": "exact",
        "ESMFOLD2_VARIANT": "full_msa",
    }


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("protenix", {"PROTENIX_OPT": "exact"}),
        ("chai1", {"CHAI1_OPT": "exact"}),
        ("esmfold2", {"ESMFOLD2_OPT": "exact"}),
        ("opendde", {"OPENDDE_OPT": "exact"}),
        ("rf3", {"ROSETTAFOLD3_OPT": "exact"}),
    ],
)
def test_library_backend_environment(fake_source, backend, expected) -> None:
    manifest, source = fake_source
    assert optimized.provider_environment(
        backend, "exact", kit_root=source
    ) == expected


def test_apptainer_builder_choices_come_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = ROOT / "scripts" / "containers" / "anthropic_kits_apptainer.py"
    spec = importlib.util.spec_from_file_location("_anthropic_kits_apptainer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    marker = {
        "source": {"revision": "a" * 40},
        "kits": {"manifest_only_kit": {"softwareIds": ["fixture"]}},
    }
    monkeypatch.setattr(module, "load_manifest", lambda: marker)
    assert module.available_kits() == ("manifest_only_kit",)
