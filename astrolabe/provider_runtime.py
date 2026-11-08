"""Translate daemon provider configuration into pinned runtime activation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Literal

from astrolabe.backend_catalog import get_backend
from astrolabe.optimized_backends import (
    KitSelection,
    OptimizedBackendError,
    UnsupportedActivationError,
    UnknownBackendError,
    activate_provider,
    build_kit_selection,
)


PROVIDER_ENV = "ASTROLABE_MODEL_PROVIDERS"
PROVIDER_REQUEST_KEY = "_provider_request"


class ProviderRuntimeError(OptimizedBackendError):
    """Provider configuration or child activation is invalid."""


class ProviderConfigError(ProviderRuntimeError):
    """The declarative provider configuration is malformed."""


ProviderName = Literal["native", "anthropic"]


@dataclass(frozen=True)
class ProviderConfig:
    provider: ProviderName
    mode: str
    model_variant: str | None = None
    kit_root: str | None = None


def prepare_provider_spec(spec: Any) -> Any:
    """Validate provider selection and prepare child startup without activating it."""
    config = _provider_config(spec)
    if config is None:
        return spec

    backend = _backend_name(getattr(spec, "backend", None))
    if config.provider == "native":
        _validate_native_config(config, backend)
        request = _native_request(backend)
        startup_env: dict[str, str] = {}
    else:
        selection = build_kit_selection(
            backend,
            config.mode,
            config.model_variant,
            kit_root=config.kit_root,
            verify_source=False,
        )
        startup_env = dict(selection.env)
        request = _selection_request(config, selection)

    settings = dict(getattr(spec, "settings", None) or {})
    settings[PROVIDER_REQUEST_KEY] = request
    environment = dict(getattr(spec, "env", None) or {})
    environment.update(startup_env)
    changes: dict[str, Any] = {
        "settings": settings,
        "env": environment,
    }
    if hasattr(spec, "provider"):
        changes["provider"] = _config_mapping(config)
    return replace(spec, **changes)


def activate_child_provider(spec: Any) -> dict[str, Any]:
    """Activate one prepared provider before the child imports its model factory."""
    request = _prepared_request(spec)
    config = _provider_config(spec)
    if config is None:
        if request is None:
            backend = _backend_name(getattr(spec, "backend", None))
            return _activation_event(
                backend=backend,
                provider="native",
                kit=None,
                mode="off",
                revision=None,
                model_identity=f"astrolabe-native:{backend}",
                state="native",
                active=False,
                report={},
            )
        config = _config_from_request(request)

    backend = _backend_name(getattr(spec, "backend", None))
    if config.provider == "native":
        _validate_native_config(config, backend)
        expected = _native_request(backend)
        if request is None:
            raise ProviderConfigError(
                "explicit native provider must pass through prepare_provider_spec "
                "before child activation"
            )
        _require_request_match(request, expected)
        return _activation_event(
            backend=backend,
            provider="native",
            kit=None,
            mode="off",
            revision=None,
            model_identity=expected["model_identity"],
            state="native",
            active=False,
            report={},
        )

    selection = build_kit_selection(
        backend,
        config.mode,
        config.model_variant,
        kit_root=config.kit_root,
        verify_source=True,
    )
    expected = _selection_request(config, selection)
    if request is None:
        raise ProviderConfigError(
            "Anthropic provider must pass through prepare_provider_spec before "
            "child activation"
        )
    _require_request_match(request, expected)
    _require_daemon_capability(selection)
    report = activate_provider(selection)
    if selection.mode != "off" and not report.active:
        raise UnsupportedActivationError(
            f"{selection.spec.backend} returned an inactive provider report; "
            "the daemon will not fall back"
        )
    return _activation_event(
        backend=selection.spec.backend,
        provider="anthropic",
        kit=report.kit,
        mode=report.mode,
        revision=report.revision,
        model_identity=report.model_identity,
        state=report.state,
        active=report.active,
        report=_jsonable(dict(report.provider_report)),
        activation_api=report.activation_api,
        kit_root=config.kit_root,
    )


def _provider_config(spec: Any) -> ProviderConfig | None:
    backend = getattr(spec, "backend", None)
    direct = getattr(spec, "provider", None)
    if direct is not None:
        config = _config_for_backend(direct, backend, source="DaemonSpec.provider")
        if config is None:
            raise ProviderConfigError(
                f"DaemonSpec.provider has no entry for backend {backend!r}"
            )
        return config

    environment = os.environ.copy()
    environment.update(
        {
            str(key): str(value)
            for key, value in (getattr(spec, "env", None) or {}).items()
        }
    )
    raw = environment.get(PROVIDER_ENV)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ProviderConfigError(
            f"{PROVIDER_ENV} must contain a JSON backend-to-config mapping: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ProviderConfigError(
            f"{PROVIDER_ENV} must contain a JSON backend-to-config mapping"
        )
    return _config_for_backend(value, backend, source=PROVIDER_ENV)


def _config_for_backend(
    value: Any,
    backend: Any,
    *,
    source: str,
) -> ProviderConfig | None:
    if isinstance(value, ProviderConfig):
        value = asdict(value)
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ProviderConfigError(f"{source} must be a provider config mapping")

    if "provider" in value:
        config_value = value
    else:
        exact = value.get(str(backend))
        canonical: str | None = None
        try:
            canonical = get_backend(str(backend)).name
        except (TypeError, ValueError):
            pass
        config_value = exact if exact is not None else value.get(canonical)
        if config_value is None:
            return None
    if not isinstance(config_value, Mapping):
        raise ProviderConfigError(
            f"{source} entry for backend {backend!r} must be a mapping"
        )
    unknown = set(config_value) - {
        "provider",
        "mode",
        "model_variant",
        "kit_root",
    }
    if unknown:
        raise ProviderConfigError(
            f"{source} provider config has unknown field(s): "
            f"{', '.join(sorted(str(item) for item in unknown))}"
        )
    provider_value = str(config_value.get("provider", "")).strip().lower()
    if provider_value not in {"native", "anthropic"}:
        raise ProviderConfigError(
            f"{source} provider must be 'native' or 'anthropic'; got {provider_value!r}"
        )
    provider: ProviderName = "native" if provider_value == "native" else "anthropic"
    default_mode = "off" if provider == "native" else "exact"
    mode = str(config_value.get("mode", default_mode)).strip().lower()
    variant_value = config_value.get("model_variant")
    if variant_value is not None and (
        not isinstance(variant_value, str) or not variant_value.strip()
    ):
        raise ProviderConfigError(
            f"{source} model_variant must be a non-empty string or null"
        )
    root_value = config_value.get("kit_root")
    if root_value is not None:
        if not isinstance(root_value, (str, os.PathLike)) or not str(root_value).strip():
            raise ProviderConfigError(
                f"{source} kit_root must be a non-empty path string or null"
            )
        root_value = str(Path(root_value).expanduser().resolve())
    return ProviderConfig(
        provider=provider,
        mode=mode,
        model_variant=variant_value.strip() if variant_value else None,
        kit_root=root_value,
    )


def _config_from_request(request: Mapping[str, Any]) -> ProviderConfig:
    fields = {
        key: request[key]
        for key in ("provider", "mode", "model_variant", "kit_root")
        if key in request
    }
    config = _config_for_backend(fields, None, source=PROVIDER_REQUEST_KEY)
    if config is None:
        raise ProviderConfigError(f"{PROVIDER_REQUEST_KEY} has no provider config")
    return config


def _prepared_request(spec: Any) -> Mapping[str, Any] | None:
    settings = getattr(spec, "settings", None)
    if not isinstance(settings, Mapping):
        return None
    raw = settings.get(PROVIDER_REQUEST_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProviderConfigError(f"{PROVIDER_REQUEST_KEY} must be a mapping")
    return raw


def _backend_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError("daemon backend must be a non-empty string")
    try:
        return get_backend(value).name
    except ValueError as exc:
        raise UnknownBackendError(str(exc)) from exc


def _validate_native_config(config: ProviderConfig, backend: str) -> None:
    if config.mode != "off":
        raise ProviderConfigError(
            f"native provider for {backend} supports only mode='off'; "
            f"got {config.mode!r}"
        )
    if config.model_variant is not None:
        raise ProviderConfigError(
            f"native provider for {backend} does not accept model_variant"
        )
    if config.kit_root is not None:
        raise ProviderConfigError(
            f"native provider for {backend} does not accept kit_root"
        )


def _native_request(backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "provider": "native",
        "kit": None,
        "mode": "off",
        "revision": None,
        "model_identity": f"astrolabe-native:{backend}",
        "model_variant": None,
        "kit_root": None,
        "activation_api": None,
    }


def _selection_request(
    config: ProviderConfig,
    selection: KitSelection,
) -> dict[str, Any]:
    return {
        "backend": selection.spec.backend,
        "provider": config.provider,
        "kit": selection.spec.name,
        "mode": selection.mode,
        "revision": selection.revision,
        "model_identity": selection.model_identity,
        "model_variant": config.model_variant,
        "kit_root": config.kit_root,
        "activation_api": asdict(selection.activation_api),
    }


def _require_request_match(
    request: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    required = (
        "backend",
        "provider",
        "kit",
        "mode",
        "revision",
        "model_identity",
        "activation_api",
    )
    mismatched = [
        key
        for key in required
        if _jsonable(request.get(key)) != _jsonable(expected.get(key))
    ]
    if mismatched:
        raise ProviderConfigError(
            "prepared provider request no longer matches the pinned manifest or "
            f"source selection ({', '.join(mismatched)}); rerun provider preparation"
        )


def _require_daemon_capability(selection: KitSelection) -> None:
    api = selection.activation_api
    if (
        api.kind != "library"
        or api.adapter_scope != "exact"
        or not api.daemon_model_compatible
    ):
        raise UnsupportedActivationError(
            f"{selection.spec.backend} cannot use {selection.spec.name} in the "
            f"current persistent daemon: {api.note}"
        )


def _activation_event(
    *,
    backend: str,
    provider: str,
    kit: str | None,
    mode: str,
    revision: str | None,
    model_identity: str,
    state: str,
    active: bool,
    report: Mapping[str, Any],
    activation_api: str | None = None,
    kit_root: str | None = None,
) -> dict[str, Any]:
    return _jsonable(
        {
            "active": active,
            "provenance": {
                "backend": backend,
                "provider": provider,
                "kit": kit,
                "mode": mode,
                "revision": revision,
                "model_identity": model_identity,
                "state": state,
                "activation_api": activation_api,
                "kit_root": kit_root,
            },
            "report": dict(report),
        }
    )


def _config_mapping(config: ProviderConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "mode": config.mode,
        "model_variant": config.model_variant,
        "kit_root": config.kit_root,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise ProviderConfigError(
        f"provider activation report is not JSON-compatible: "
        f"{type(value).__name__}"
    )
