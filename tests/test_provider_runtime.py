from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from astrolabe import optimized_backends as optimized
from astrolabe import provider_runtime as runtime


@dataclass
class Spec:
    backend: str
    settings: dict[str, Any]
    env: dict[str, str] | None = None
    provider: dict[str, Any] | None = None


@dataclass
class LegacySpec:
    backend: str
    settings: dict[str, Any]
    env: dict[str, str] | None = None


def _provider_map(backend: str, **config: Any) -> str:
    return json.dumps({backend: config})


def _fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    manifest = optimized.load_manifest()
    root = tmp_path / "anthropic-kits"
    root.mkdir()
    (root / ".git").mkdir()
    for kit in optimized.kit_names(manifest):
        (root / kit / "opt").mkdir(parents=True)
        (root / kit / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (root / kit / "opt" / "pyproject.toml").write_text(
            "[project]\nname = 'fixture'\n", encoding="utf-8"
        )
    revision = manifest["source"]["revision"]

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return str(root)
        if args == ("rev-parse", "HEAD"):
            return revision
        raise AssertionError(f"unexpected git arguments: {args}")

    monkeypatch.setattr(optimized, "_git_output", fake_git)
    optimized.reset_activation_state()
    return root


def test_native_default_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.PROVIDER_ENV, raising=False)
    spec = Spec("protenix", {"sample": 1})

    assert runtime.prepare_provider_spec(spec) is spec


def test_optimized_mode_defaults_to_exact() -> None:
    default = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={"provider": "anthropic"},
        )
    )
    fast = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={"provider": "anthropic", "mode": "fast"},
        )
    )

    assert default.env["PROTENIX_OPT"] == "exact"
    assert default.settings[runtime.PROVIDER_REQUEST_KEY]["mode"] == "exact"
    assert fast.env["PROTENIX_OPT"] == "fast"
    assert fast.settings[runtime.PROVIDER_REQUEST_KEY]["mode"] == "fast"


def test_env_map_is_validated_without_reading_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimized,
        "verify_source_checkout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("coordinator read provider source")
        ),
    )
    spec = Spec(
        "protenix",
        {},
        env={
            runtime.PROVIDER_ENV: _provider_map(
                "protenix", provider="anthropic", mode="exact"
            ),
            "EXISTING": "kept",
        },
    )

    prepared = runtime.prepare_provider_spec(spec)

    assert spec.env == {
        runtime.PROVIDER_ENV: _provider_map(
            "protenix", provider="anthropic", mode="exact"
        ),
        "EXISTING": "kept",
    }
    assert prepared.env["EXISTING"] == "kept"
    assert prepared.env["PROTENIX_OPT"] == "exact"
    assert "CHAI1_OPT" not in prepared.env
    request = json.loads(json.dumps(prepared.settings[runtime.PROVIDER_REQUEST_KEY]))
    assert request["kit"] == "protenix_v2"
    assert request["mode"] == "exact"
    assert request["revision"] == optimized.load_manifest()["source"]["revision"]
    assert request["activation_api"]["module"] == "protenix_opt"


def test_preparation_supports_spec_without_provider_field() -> None:
    spec = LegacySpec(
        "protenix",
        {},
        env={
            runtime.PROVIDER_ENV: _provider_map(
                "protenix", provider="anthropic", mode="exact"
            )
        },
    )

    prepared = runtime.prepare_provider_spec(spec)

    assert prepared.env["PROTENIX_OPT"] == "exact"
    assert runtime.PROVIDER_REQUEST_KEY in prepared.settings


def test_direct_provider_takes_precedence_and_native_is_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        runtime.PROVIDER_ENV,
        _provider_map("protenix", provider="anthropic", mode="fast"),
    )
    spec = Spec(
        "protenix",
        {},
        provider={"provider": "native", "mode": "off"},
    )

    prepared = runtime.prepare_provider_spec(spec)
    event = runtime.activate_child_provider(prepared)

    assert prepared.provider == {
        "provider": "native",
        "mode": "off",
        "model_variant": None,
        "kit_root": None,
    }
    assert "PROTENIX_OPT" not in (prepared.env or {})
    assert event["active"] is False
    assert event["provenance"]["state"] == "native"
    assert event["provenance"]["provider"] == "native"


def test_anthropic_off_is_upstream_stock_and_never_imports_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    spec = Spec(
        "protenix",
        {},
        provider={
            "provider": "anthropic",
            "mode": "off",
            "kit_root": str(source),
        },
    )
    prepared = runtime.prepare_provider_spec(spec)
    monkeypatch.setattr(
        optimized,
        "activate_provider",
        lambda selection: optimized.ActivationReport(
            active=False,
            state="upstream-stock",
            kit=selection.spec.name,
            model_identity=selection.model_identity,
            mode=selection.mode,
            revision=selection.revision,
            activation_api="off",
            provider_report={},
        ),
    )
    import_calls: list[str] = []
    monkeypatch.setattr(
        optimized.importlib,
        "import_module",
        lambda name: import_calls.append(name),
    )

    event = runtime.activate_child_provider(prepared)

    assert event["active"] is False
    assert event["provenance"]["state"] == "upstream-stock"
    assert import_calls == []


def test_exact_calls_active_provider_once_with_requested_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    spec = Spec(
        "protenix",
        {},
        provider={
            "provider": "anthropic",
            "mode": "exact",
            "kit_root": str(source),
        },
    )
    prepared = runtime.prepare_provider_spec(spec)
    calls: list[str] = []

    def enable(mode: str, *, strict: bool) -> dict[str, Any]:
        assert strict is True
        calls.append(mode)
        return {"active": True, "mode": mode, "applied": "applied"}

    monkeypatch.setitem(
        sys.modules,
        "protenix_opt",
        SimpleNamespace(enable=enable),
    )
    first = runtime.activate_child_provider(prepared)
    second = runtime.activate_child_provider(prepared)

    assert calls == ["exact"]
    assert first == second
    assert first["active"] is True
    assert first["provenance"] == {
        "backend": "protenix",
        "provider": "anthropic",
        "kit": "protenix_v2",
        "mode": "exact",
        "revision": optimized.load_manifest()["source"]["revision"],
        "model_identity": first["provenance"]["model_identity"],
        "state": "applied",
        "activation_api": "protenix_opt.enable",
        "kit_root": str(source),
    }
    assert first["report"] == {
        "active": True,
        "mode": "exact",
        "applied": "applied",
    }


def test_missing_provider_module_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    prepared = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={
                "provider": "anthropic",
                "mode": "exact",
                "kit_root": str(source),
            },
        )
    )

    def missing(name: str) -> Any:
        if name == "protenix_opt":
            raise ModuleNotFoundError("fixture module is absent")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(optimized.importlib, "import_module", missing)

    with pytest.raises(optimized.UnsupportedActivationError, match="not importable"):
        runtime.activate_child_provider(prepared)


def test_exact_report_seam_rejects_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    prepared = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={
                "provider": "anthropic",
                "mode": "exact",
                "kit_root": str(source),
            },
        )
    )
    calls: list[str] = []

    def activate(selection: optimized.KitSelection) -> optimized.ActivationReport:
        calls.append(selection.mode)
        return optimized.ActivationReport(
            active=False,
            state="refused",
            kit=selection.spec.name,
            model_identity=selection.model_identity,
            mode=selection.mode,
            revision=selection.revision,
            activation_api="protenix_opt.enable",
            provider_report={"active": False, "mode": selection.mode},
        )

    monkeypatch.setattr(runtime, "activate_provider", activate)
    with pytest.raises(optimized.UnsupportedActivationError, match="inactive"):
        runtime.activate_child_provider(prepared)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("active", False, "did not activate"),
    ],
)
def test_inactive_report_fails_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    expected: str,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    prepared = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={
                "provider": "anthropic",
                "mode": "exact",
                "kit_root": str(source),
            },
        )
    )
    selection = optimized.build_kit_selection(
        "protenix", "exact", kit_root=source
    )

    monkeypatch.setitem(
        sys.modules,
        "protenix_opt",
        type(
            "Provider",
            (),
            {
                "enable": lambda self, mode, *, strict: {
                    field: value,
                    "mode": mode,
                    "reason": "fixture gate inactive",
                }
            },
        )(),
    )

    with pytest.raises(optimized.ActivationNotActiveError, match=expected):
        optimized.activate_provider(selection)
    with pytest.raises(optimized.ActivationNotActiveError, match=expected):
        runtime.activate_child_provider(prepared)


def test_deferred_report_is_not_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    selection = optimized.build_kit_selection(
        "protenix", "exact", kit_root=source
    )
    monkeypatch.setitem(
        sys.modules,
        "protenix_opt",
        type(
            "Provider",
            (),
            {
                "enable": lambda self, mode, *, strict: {
                    "active": True,
                    "applied": "deferred",
                    "mode": mode,
                }
            },
        )(),
    )

    with pytest.raises(optimized.ActivationNotActiveError, match="deferred"):
        optimized.activate_provider(selection)


def test_provider_error_preserves_type_and_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    prepared = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={
                "provider": "anthropic",
                "mode": "exact",
                "kit_root": str(source),
            },
        )
    )
    expected = optimized.ActivationNotActiveError("fixture provider failure")

    def fail(_selection: optimized.KitSelection) -> optimized.ActivationReport:
        raise expected

    monkeypatch.setattr(runtime, "activate_provider", fail)
    with pytest.raises(optimized.ActivationNotActiveError) as raised:
        runtime.activate_child_provider(prepared)

    assert raised.value is expected


def test_source_revision_drift_is_rejected_in_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    prepared = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={
                "provider": "anthropic",
                "mode": "exact",
                "kit_root": str(source),
            },
        )
    )
    monkeypatch.setattr(
        optimized,
        "_git_output",
        lambda _root, *args: str(source)
        if args == ("rev-parse", "--show-toplevel")
        else "0" * 40,
    )

    with pytest.raises(optimized.SourceRevisionMismatchError, match="revision drift"):
        runtime.activate_child_provider(prepared)


def test_changed_provider_semantics_are_rejected_in_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    prepared = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={
                "provider": "anthropic",
                "mode": "exact",
                "kit_root": str(source),
            },
        )
    )
    request = dict(prepared.settings[runtime.PROVIDER_REQUEST_KEY])
    request["model_identity"] = "different-model@different-revision"
    prepared = replace(
        prepared,
        settings={
            **prepared.settings,
            runtime.PROVIDER_REQUEST_KEY: request,
        },
    )

    with pytest.raises(runtime.ProviderConfigError, match="no longer matches"):
        runtime.activate_child_provider(prepared)


def test_empty_source_is_actionable_in_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "empty-source"
    source.mkdir()
    prepared = runtime.prepare_provider_spec(
        Spec(
            "protenix",
            {},
            provider={
                "provider": "anthropic",
                "mode": "exact",
                "kit_root": str(source),
            },
        )
    )

    with pytest.raises(optimized.SourceUnavailableError, match="empty.*independent"):
        runtime.activate_child_provider(prepared)


@pytest.mark.parametrize(
    ("backend", "variant", "message"),
    [
        ("boltz", None, "Boltz2"),
        ("esmfold", None, "ESMFold2"),
        ("alphafold3", None, "explicit model_variant"),
    ],
)
def test_incompatible_model_variants_fail_during_preparation(
    backend: str,
    variant: str | None,
    message: str,
) -> None:
    spec = Spec(
        backend,
        {},
        provider={
            "provider": "anthropic",
            "mode": "exact",
            "model_variant": variant,
        },
    )

    with pytest.raises(optimized.ProviderMismatchError, match=message):
        runtime.prepare_provider_spec(spec)


@pytest.mark.parametrize("backend", ["chai1", "esmfold2", "opendde", "esmif", "alphafold3"])
def test_unsupported_or_outside_adapter_is_refused_before_hook(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    mode = "fast" if backend == "esmif" else "exact"
    variant = "p2" if backend == "alphafold3" else (
        "full_msa" if backend == "esmfold2" else None
    )
    prepared = runtime.prepare_provider_spec(
        Spec(
            backend,
            {},
            provider={
                "provider": "anthropic",
                "mode": mode,
                "model_variant": variant,
                "kit_root": str(source),
            },
        )
    )
    monkeypatch.setattr(
        optimized,
        "activate_provider",
        lambda _selection: (_ for _ in ()).throw(
            AssertionError("unsupported adapter reached hook")
        ),
    )

    with pytest.raises(optimized.UnsupportedActivationError):
        runtime.activate_child_provider(prepared)


def test_unknown_backend_fails() -> None:
    with pytest.raises(optimized.UnknownBackendError):
        runtime.prepare_provider_spec(
            Spec(
                "not-a-backend",
                {},
                provider={"provider": "anthropic", "mode": "exact"},
            )
        )


def test_unknown_mode_fails() -> None:
    with pytest.raises(optimized.UnknownModeError, match="not declared"):
        runtime.prepare_provider_spec(
            Spec("protenix", {}, provider={"provider": "anthropic", "mode": "turbo"})
        )


def test_native_nonoff_fails() -> None:
    with pytest.raises(runtime.ProviderConfigError, match="only mode='off'"):
        runtime.prepare_provider_spec(
            Spec("protenix", {}, provider={"provider": "native", "mode": "exact"})
        )


def test_malformed_provider_environment_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = Spec("protenix", {}, env={runtime.PROVIDER_ENV: "{"})
    with pytest.raises(runtime.ProviderConfigError, match="JSON"):
        runtime.prepare_provider_spec(spec)


def test_helper_import_does_not_load_torch_or_jax() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, astrolabe.provider_runtime; "
            "assert not {'torch', 'jax', 'jaxlib'} & set(sys.modules)",
        ],
        cwd=root,
        check=True,
    )


def test_real_child_seam_prepares_env_before_factory_and_records_report(
    tmp_path: Path,
) -> None:
    import dataclasses

    original_dataclass = dataclasses.dataclass
    if sys.version_info < (3, 10):
        def compatible_dataclass(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("slots", None)
            return original_dataclass(*args, **kwargs)

        dataclasses.dataclass = compatible_dataclass
    try:
        from astrolabe.model_daemon import DaemonSpec, OwnedModelDaemon
    finally:
        dataclasses.dataclass = original_dataclass

    wrapper = tmp_path / "daemon-python"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import dataclasses, runpy, sys\n"
        "_dataclass = dataclasses.dataclass\n"
        "def _compatible(*args, **kwargs):\n"
        "    kwargs.pop('slots', None)\n"
        "    return _dataclass(*args, **kwargs)\n"
        "dataclasses.dataclass = _compatible\n"
        "sys.argv = ['astrolabe.model_daemon', *sys.argv[4:]]\n"
        "runpy.run_module('astrolabe.model_daemon', run_name='__main__', alter_sys=True)\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    module_root = tmp_path / "modules"
    module_root.mkdir()
    (module_root / "protenix_opt.py").write_text(
        "STATE = 'unactivated'\n"
        "def enable(mode, *, strict=False, trigger=None):\n"
        "    global STATE\n"
        "    STATE = 'fixture-active'\n"
        "    return {'active': True, 'mode': mode, 'applied': 'fixture-active'}\n",
        encoding="utf-8",
    )
    (module_root / "fixture_model.py").write_text(
        "import os\n"
        "import protenix_opt\n"
        "def factory(settings):\n"
        "    observed = (protenix_opt.STATE, os.environ.get('PROTENIX_OPT'))\n"
        "    return lambda payload: {'observed': observed}\n",
        encoding="utf-8",
    )

    source = tmp_path / "kit-source"
    source.mkdir()
    for kit in optimized.kit_names():
        (source / kit / "opt").mkdir(parents=True)
        (source / kit / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (source / kit / "opt" / "pyproject.toml").write_text(
            "[project]\nname = 'fixture'\n", encoding="utf-8"
        )
    (source / ".git").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *--show-toplevel*) printf '%s\\n' \"$FAKE_SOURCE_ROOT\" ;;\n"
        "  *HEAD*) printf '%s\\n' \"$FAKE_SOURCE_REVISION\" ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    revision = optimized.load_manifest()["source"]["revision"]

    provider_map = _provider_map(
        "protenix",
        provider="anthropic",
        mode="exact",
        kit_root=str(source),
    )
    spec = DaemonSpec(
        backend="protenix",
        factory="fixture_model:factory",
        settings={},
        python_bin=str(wrapper),
        env={
            runtime.PROVIDER_ENV: provider_map,
            "PYTHONPATH": f"{module_root}",
            "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "FAKE_SOURCE_ROOT": str(source),
            "FAKE_SOURCE_REVISION": revision,
        },
    )
    owner = OwnedModelDaemon(spec, startup_timeout=20)
    try:
        result = owner.request({})
        log = Path(owner.log_path).read_text(encoding="utf-8")
    finally:
        owner.shutdown()

    assert result == {"observed": ("fixture-active", "exact")}
    assert '"event": "provider_activated"' in log
    assert '"state": "fixture-active"' in log
