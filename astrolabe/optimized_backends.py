"""Select and activate pinned Anthropic optimization kits.

The kit inventory and declared modes come from ``config/anthropic_kits.yaml``.
This module contains only the provider-specific API mapping needed to connect a
manifest entry to a real in-process hook or to an explicitly unsupported path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import importlib
from pathlib import Path
import subprocess
import threading
from typing import Any, Literal

from omegaconf import OmegaConf

from astrolabe.backend_catalog import get_backend


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "anthropic_kits.yaml"
VALID_MODES = frozenset({"off", "exact", "fast", "big"})


class OptimizedBackendError(RuntimeError):
    """Base error for provider selection and activation."""


class UnknownBackendError(OptimizedBackendError):
    """The requested Astrolabe backend is unknown."""


class UnknownModeError(OptimizedBackendError):
    """The requested mode is not declared for the selected kit."""


class NativeBackendOnlyError(OptimizedBackendError):
    """The manifest declares no semantic provider for this backend."""


class ProviderMismatchError(OptimizedBackendError):
    """The kit model identity is incompatible with the selected Astrolabe backend."""


class SourceUnavailableError(OptimizedBackendError):
    """The pinned kit source is missing or is not an independent checkout."""


class SourceRevisionMismatchError(OptimizedBackendError):
    """The selected kit source is not at the manifest revision."""


class ActivationNotActiveError(OptimizedBackendError):
    """A provider returned without applying its requested optimization."""


class UnsupportedActivationError(OptimizedBackendError):
    """The provider cannot be activated in the current Astrolabe process."""


class ProviderPoisonedError(OptimizedBackendError):
    """A hook may have changed this process; only a fresh process is safe."""


class ActivationConflictError(OptimizedBackendError):
    """A second, incompatible activation was requested for one provider."""


ActivationKind = Literal["library", "subprocess", "unsupported"]
AdapterScope = Literal["exact", "partial", "outside", "incompatible"]


@dataclass(frozen=True)
class ActivationAPI:
    kind: ActivationKind
    module: str | None
    function: str | None
    startup_env: str | None
    adapter_scope: AdapterScope
    daemon_model_compatible: bool
    note: str


@dataclass(frozen=True)
class KitSpec:
    name: str
    backend: str
    software_ids: tuple[str, ...]
    role: str
    modes: tuple[str, ...]
    install_args: tuple[str, ...]
    weights_env: str | None
    interpreter_contract: str | None


@dataclass(frozen=True)
class KitSelection:
    spec: KitSpec
    model_identity: str
    mode: str
    revision: str
    env: Mapping[str, str]
    activation_api: ActivationAPI


@dataclass(frozen=True)
class ActivationReport:
    active: bool
    state: str
    kit: str
    model_identity: str
    mode: str
    revision: str
    activation_api: str
    provider_report: Mapping[str, Any]


@dataclass(frozen=True)
class _ProviderRule:
    software_id: str
    model_identity: str
    activation_api: ActivationAPI
    mismatch: str | None = None


_PROVIDER_RULES: dict[str, _ProviderRule] = {
    "alphafold3": _ProviderRule(
        "alphafold3",
        "alphafold3-of3-p2:v3.1.4@bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997",
        ActivationAPI(
            "subprocess",
            None,
            None,
            "AF3_JAX_OPT",
            "outside",
            False,
            "The loaded ModelRunner cannot consume this route. Use af3-jax-opt pred "
            "or run.sh pred; its OF3-converted P2 weights are not a licensed native "
            "AlphaFold 3 replacement.",
        ),
    ),
    "protenix": _ProviderRule(
        "protenix",
        "protenix-v2:2.0.0@2475421477ab414b571149ad4a875c390ff8a35d",
        ActivationAPI(
            "library",
            "protenix_opt",
            "enable",
            "PROTENIX_OPT",
            "exact",
            True,
            "Enable before ProtenixRunner constructs Protenix; ProtenixRunner.predict "
            "then reuses the loaded model.",
        ),
    ),
    "boltz": _ProviderRule(
        "boltz",
        "boltz2:2.2.1@cb04aeccdd480fd4db707f0bbafde538397fa2ac",
        ActivationAPI(
            "unsupported",
            None,
            None,
            "BOLTZ2_OPT",
            "incompatible",
            False,
            "The manifest provider is Boltz 2. Astrolabe's Boltz1Runner loads "
            "Boltz1 and boltz1_conf.ckpt; the kit requires its boltz2-opt worker.",
        ),
        mismatch=(
            "Boltz2 is not a replacement for Astrolabe's Boltz1 runner; use a "
            "separate Boltz2 adapter and the boltz2-opt worker."
        ),
    ),
    "chai1": _ProviderRule(
        "chai1",
        "chai1:0.6.1@8d5ac0f93e9b6ea4c3a6545c253a6381c0f3694b",
        ActivationAPI(
            "library",
            "chai1_opt",
            "enable",
            "CHAI1_OPT",
            "exact",
            False,
            "The hook patches chai_lab.chai1, which the evaluator calls, but upstream "
            "run_inference reloads its TorchScript components on every invocation.",
        ),
    ),
    "esmif": _ProviderRule(
        "esmif",
        "esm-if1:2.0.1@2b369911bb5b4b0dda914521b9475cad1656b2ac",
        ActivationAPI(
            "subprocess",
            None,
            None,
            "ESM_IF1_OPT",
            "outside",
            False,
            "The fast kit batches sampling in its own child. It does not optimize "
            "the probability-logits path used by ESMIF1Mutator.",
        ),
    ),
    "esmfold": _ProviderRule(
        "esmfold",
        "esmfold2:esm-3.3.0@26b0bc2b771e3e419ea74f445a5f35cc094a1509",
        ActivationAPI(
            "unsupported",
            None,
            None,
            "ESMFOLD2_OPT",
            "incompatible",
            False,
            "The manifest provider is Biohub ESMFold2, not Astrolabe's ESMFold v1.",
        ),
        mismatch=(
            "ESMFold2 is a different model family from the current ESMFold backend; "
            "select the explicit esmfold2 backend for a new adapter."
        ),
    ),
    "esmfold2": _ProviderRule(
        "esmfold2",
        "esmfold2:esm-3.3.0@26b0bc2b771e3e419ea74f445a5f35cc094a1509/"
        "transformers-4.57.6@ef32577f55da19a4989cd7b22e004dc43a4998cb",
        ActivationAPI(
            "library",
            "esmfold2_opt",
            "enable",
            "ESMFOLD2_OPT",
            "outside",
            True,
            "Enable before ESMFold2Model construction; the kit server applies its "
            "configuration on that model's first fold. The current Astrolabe worker "
            "starts a fresh subprocess for every batch.",
        ),
    ),
    "opendde": _ProviderRule(
        "opendde",
        "opendde:1.1.1@ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f",
        ActivationAPI(
            "library",
            "opendde_opt",
            "enable",
            "OPENDDE_OPT",
            "outside",
            True,
            "The hook wraps runner.batch_inference.get_default_runner. Astrolabe's "
            "current adapter starts runner.batch_inference in a separate process.",
        ),
    ),
    "rf3": _ProviderRule(
        "rf3",
        "rosettafold3:foundry@4010e3e2e7350edada3e25a45c908c6bf407df4d",
        ActivationAPI(
            "library",
            "rosettafold3_opt",
            "enable",
            "ROSETTAFOLD3_OPT",
            "exact",
            True,
            "Enable before rf3.graph_flags is imported. RF3RunnerService caches the "
            "configured inference engine for later requests, but the immediate "
            "enable report is deferred until that import; publish active only after "
            "a qualified configured report.",
        ),
    ),
}


_ACTIVATION_LOCK = threading.RLock()
_ACTIVATION_CACHE: dict[tuple[Any, ...], ActivationReport] = {}
_PROCESS_ACTIVE: dict[str, tuple[Any, ...]] = {}
_PROCESS_STATES: dict[str, str] = {}
_PROCESS_FAILURES: dict[str, BaseException] = {}


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the single Anthropic kit manifest."""
    manifest_path = Path(path or MANIFEST_PATH).expanduser().resolve()
    value = OmegaConf.to_container(OmegaConf.load(str(manifest_path)), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"invalid Anthropic kit manifest: {manifest_path}")
    source = value.get("source")
    kits = value.get("kits")
    if not isinstance(source, dict) or not isinstance(kits, dict):
        raise ValueError(f"invalid Anthropic kit manifest: {manifest_path}")
    revision = source.get("revision")
    source_path = source.get("path")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Anthropic kit manifest has no source revision: {manifest_path}")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError(f"Anthropic kit manifest has no source path: {manifest_path}")
    for name, spec in kits.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError(f"invalid Anthropic kit entry in {manifest_path}")
        _validate_spec(name, spec)
    return value


def kit_names(manifest: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return kit names in manifest order."""
    data = manifest or load_manifest()
    return tuple(str(name) for name in data["kits"])


def selected_kit_specs(
    manifest: Mapping[str, Any],
    names: Iterable[str] | None = None,
    software_ids: Iterable[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Select manifest entries without maintaining a second kit list."""
    kits = manifest["kits"]
    requested_names = set(names or ())
    requested_software = set(software_ids or ())
    unknown = requested_names - set(kits)
    if unknown:
        raise ValueError(f"unknown Anthropic kit(s): {', '.join(sorted(unknown))}")
    rows: list[tuple[str, dict[str, Any]]] = []
    for name, raw_spec in kits.items():
        ids = set(raw_spec.get("softwareIds", ()))
        if requested_names and name not in requested_names:
            continue
        if requested_software and not ids.intersection(requested_software):
            continue
        rows.append((str(name), dict(raw_spec)))
    if not rows:
        raise ValueError("no Anthropic kits selected")
    return rows


def verify_source_checkout(
    source_root: str | Path,
    revision: str,
) -> Path:
    """Require an independent top-level checkout at the manifest revision."""
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise SourceUnavailableError(
            f"Anthropic kit source is missing at {root}; initialize the pinned source checkout"
        )
    try:
        has_entries = any(root.iterdir())
    except OSError as exc:
        raise SourceUnavailableError(f"cannot read Anthropic kit source at {root}: {exc}") from exc
    if not has_entries:
        raise SourceUnavailableError(
            f"Anthropic kit source is empty at {root}; initialize its independent .git checkout"
        )
    if not (root / ".git").exists():
        raise SourceUnavailableError(
            f"Anthropic kit source at {root} has no top-level .git; "
            "an inherited parent repository revision is not acceptable"
        )
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise SourceUnavailableError(
            f"Anthropic kit source at {root} belongs to another repository at {top_level}; "
            "initialize the vendor checkout"
        )
    actual = _git_output(root, "rev-parse", "HEAD")
    if actual != revision:
        raise SourceRevisionMismatchError(
            f"Anthropic kit source revision drift: expected {revision}, observed {actual}"
        )
    return root


def validate_manifest_source(
    manifest: Mapping[str, Any] | None = None,
    *,
    source_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Validate the source checkout and every manifest kit directory."""
    data = dict(manifest or load_manifest())
    source = data["source"]
    root = verify_source_checkout(
        source_root or ROOT / str(source["path"]),
        str(source["revision"]),
    )
    rows: list[dict[str, Any]] = []
    for name, spec in selected_kit_specs(data):
        _validate_kit_tree(root, name)
        rows.append(
            {
                "kit": name,
                "softwareIds": list(spec.get("softwareIds", ())),
                "role": spec.get("role"),
                "modes": list(spec.get("modes", ())),
                "run": str(root / name / "run.sh"),
            }
        )
    return rows


def build_kit_selection(
    backend: str,
    mode: str = "exact",
    model_variant: str | None = None,
    *,
    kit_root: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    verify_source: bool = True,
) -> KitSelection:
    """Validate one backend/provider pairing and optionally its pinned source."""
    data = dict(manifest or load_manifest())
    try:
        backend_record = get_backend(backend)
    except ValueError as exc:
        raise UnknownBackendError(str(exc)) from exc

    rule = _PROVIDER_RULES.get(backend_record.name)
    if rule is None:
        unmapped = data.get("unmapped", {})
        if backend_record.name in unmapped:
            raise NativeBackendOnlyError(
                f"{backend_record.name} has no semantic Anthropic provider; "
                f"keep the native backend ({unmapped[backend_record.name]})"
            )
        raise UnknownBackendError(
            f"no optimized-provider rule for backend {backend_record.name!r}"
        )

    matches = [
        (name, spec)
        for name, spec in selected_kit_specs(
            data, software_ids=backend_record.software
        )
        if rule.software_id in spec.get("softwareIds", ())
    ]
    if not matches:
        raise ProviderMismatchError(
            f"no manifest kit declares software id {rule.software_id!r} "
            f"for backend {backend_record.name!r}"
        )
    kit, raw_spec = matches[0]
    if rule.mismatch:
        raise ProviderMismatchError(rule.mismatch)

    modes = tuple(str(item) for item in raw_spec.get("modes", ()))
    requested_mode = str(mode).strip().lower()
    if requested_mode not in modes:
        raise UnknownModeError(
            f"mode {mode!r} is not declared for {kit}; choose from {', '.join(modes)}"
        )

    pinned_variant = _install_arg(raw_spec, "--variant")
    requested_variant = model_variant
    if pinned_variant is not None:
        if requested_variant != pinned_variant:
            raise ProviderMismatchError(
                f"{kit} requires the explicit model_variant={pinned_variant!r}; "
                f"got {requested_variant!r}. Its converted checkpoint must not be "
                "selected as a native-model replacement."
            )
        requested_variant = pinned_variant
    elif requested_variant is not None and rule.activation_api.startup_env != "ESMFOLD2_OPT":
        raise ProviderMismatchError(
            f"{kit} does not expose a model variant selection for {backend_record.name}"
        )

    source = data["source"]
    if verify_source:
        root = verify_source_checkout(
            kit_root or ROOT / str(source["path"]),
            str(source["revision"]),
        )
        _validate_kit_tree(root, kit)

    spec = KitSpec(
        name=kit,
        backend=backend_record.name,
        software_ids=tuple(str(item) for item in raw_spec.get("softwareIds", ())),
        role=str(raw_spec.get("role") or ""),
        modes=modes,
        install_args=tuple(str(item) for item in raw_spec.get("installArgs", ())),
        weights_env=(
            str(raw_spec["weightsEnv"]) if raw_spec.get("weightsEnv") else None
        ),
        interpreter_contract=(
            str(raw_spec["interpreterContract"])
            if raw_spec.get("interpreterContract")
            else None
        ),
    )
    env: dict[str, str] = {}
    if rule.activation_api.startup_env and requested_mode != "off":
        env[rule.activation_api.startup_env] = requested_mode
    if (
        requested_mode != "off"
        and requested_variant is not None
        and rule.activation_api.startup_env == "ESMFOLD2_OPT"
    ):
        env["ESMFOLD2_VARIANT"] = str(requested_variant)

    identity = rule.model_identity
    if requested_variant is not None and requested_variant not in identity:
        identity = f"{identity}/{requested_variant}"
    return KitSelection(
        spec=spec,
        model_identity=identity,
        mode=requested_mode,
        revision=str(source["revision"]),
        env=env,
        activation_api=rule.activation_api,
    )


def provider_environment(
    backend: str,
    mode: str = "exact",
    model_variant: str | None = None,
    kit_root: str | Path | None = None,
) -> dict[str, str]:
    """Return child-startup variables for one validated provider selection."""
    selection = build_kit_selection(
        backend,
        mode,
        model_variant,
        kit_root=kit_root,
    )
    return dict(selection.env)


def activate_provider(selection: KitSelection) -> ActivationReport:
    """Call one validated provider library hook once in the current process."""
    if not isinstance(selection, KitSelection):
        raise TypeError("selection must be a KitSelection")
    api = selection.activation_api
    # Even stock/off cannot assert a clean module after an irreversible hook.
    with _ACTIVATION_LOCK:
        phase = _PROCESS_STATES.get(api.module)
        if phase == "POISONED":
            raise ProviderPoisonedError(
                f"{api.module} is POISONED; use a fresh process"
            ) from _PROCESS_FAILURES.get(api.module)
        if phase in {"APPLYING", "VERIFYING"}:
            raise ActivationConflictError(f"{api.module} activation is {phase}")
        if selection.mode == "off" and api.module in _PROCESS_ACTIVE:
            raise ActivationConflictError(f"{api.module} is already active; off cannot undo it")
        if selection.mode == "off":
            return ActivationReport(
                active=False,
                state="upstream-stock",
                kit=selection.spec.name,
                model_identity=selection.model_identity,
                mode=selection.mode,
                revision=selection.revision,
                activation_api="off",
                provider_report={},
            )
    if api.kind != "library" or not api.module or not api.function:
        raise UnsupportedActivationError(api.note)

    key = _selection_key(selection)
    with _ACTIVATION_LOCK:
        phase = _PROCESS_STATES.get(api.module)
        if phase == "POISONED":
            raise ProviderPoisonedError(f"{api.module} is POISONED; use a fresh process") from _PROCESS_FAILURES.get(api.module)
        if phase in {"APPLYING", "VERIFYING"}:
            raise ActivationConflictError(f"{api.module} activation is {phase}")
        cached = _ACTIVATION_CACHE.get(key)
        if cached is not None:
            return cached
        previous = _PROCESS_ACTIVE.get(api.module)
        if previous is not None and previous != key:
            raise ActivationConflictError(
                f"{api.module} is already active with a different mode, model, or variant"
            )
        _PROCESS_STATES[api.module] = "PREPARING"
        try:
            module = importlib.import_module(api.module)
        except ImportError as exc:
            raise UnsupportedActivationError(
                f"{api.module} is not importable; install the selected kit in this "
                f"child environment ({exc})"
            ) from exc
        hook = getattr(module, api.function, None)
        if not callable(hook):
            raise UnsupportedActivationError(
                f"{api.module}.{api.function} is absent; the selected kit is not installed"
            )
        kwargs: dict[str, Any] = {"strict": True}
        if api.startup_env == "ESMFOLD2_OPT":
            variant = selection.env.get("ESMFOLD2_VARIANT")
            if variant:
                kwargs["variant"] = variant
        try:
            _PROCESS_STATES[api.module] = "APPLYING"
            try:
                provider_report = hook(selection.mode, **kwargs)
            except Exception as exc:
                raise ActivationNotActiveError(
                    f"{api.module}.{api.function} refused {selection.mode}: {exc}"
                ) from exc
            _PROCESS_STATES[api.module] = "VERIFYING"
            if not isinstance(provider_report, Mapping):
                raise ActivationNotActiveError(
                    f"{api.module}.{api.function} returned no activation mapping"
                )
            report = dict(provider_report)
            if (
                report.get("active") is not True
                or report.get("apply_failed") is True
                or report.get("refused") is True
            ):
                reason = report.get("reason") or "provider did not set active=True"
                raise ActivationNotActiveError(
                    f"{api.module}.{api.function} did not activate {selection.mode}: {reason}"
                )
            reported_mode = report.get("mode")
            if reported_mode is not None and str(reported_mode).lower() != selection.mode:
                raise ActivationNotActiveError(
                    f"{api.module}.{api.function} activated {reported_mode!r}, "
                    f"not the requested {selection.mode!r}"
                )
            state = _qualified_activation_state(selection, report, api)
            result = ActivationReport(
                active=True,
                state=state,
                kit=selection.spec.name,
                model_identity=selection.model_identity,
                mode=selection.mode,
                revision=selection.revision,
                activation_api=f"{api.module}.{api.function}",
                provider_report=report,
            )
            _PROCESS_STATES[api.module] = "ACTIVE"
            _PROCESS_ACTIVE[api.module] = key
            _ACTIVATION_CACHE[key] = result
            return result
        except BaseException as exc:
            _PROCESS_STATES[api.module] = "POISONED"
            _PROCESS_FAILURES[api.module] = exc
            raise


def _qualified_activation_state(
    selection: KitSelection,
    report: Mapping[str, Any],
    api: ActivationAPI,
) -> str:
    applied_state = report.get("applied")
    if applied_state is None:
        applied_state = report.get("state")
    normalized_state = str(applied_state).strip().lower() if applied_state is not None else ""
    blocked_states = {
        "armed",
        "deferred",
        "dry-run",
        "dry_run",
        "failed",
        "inactive",
        "mismatch",
        "pending",
        "refused",
    }
    if normalized_state in blocked_states:
        if normalized_state == "deferred":
            reason = (
                "provider armed a deferred hook; its levers have not yet been applied"
            )
        else:
            reason = f"provider report state is {normalized_state!r}"
        raise ActivationNotActiveError(
            f"{api.module}.{api.function} did not publish active {selection.mode}: "
            f"{reason}"
        )

    applied_levers = report.get("levers_applied")
    if applied_levers is None:
        applied_levers = ()
    if not isinstance(applied_levers, (list, tuple, set, frozenset)):
        raise ActivationNotActiveError(
            f"{api.module}.{api.function} returned malformed levers_applied; "
            "the activation state is unqualified"
        )
    if not normalized_state and not applied_levers:
        raise ActivationNotActiveError(
            f"{api.module}.{api.function} returned active=True without applied "
            "lever evidence or an applied state; the activation state is unqualified"
        )

    if selection.spec.backend in {"chai1", "esmfold2"} and report.get("partial"):
        raise ActivationNotActiveError(
            f"{selection.spec.name} reported a partial activation; the requested "
            "mode does not have all of its levers applied"
        )
    if selection.spec.backend == "esmfold2":
        deferred_levers = report.get("levers_deferred")
        if (
            normalized_state != "configured"
            or not applied_levers
            or bool(deferred_levers)
        ):
            raise ActivationNotActiveError(
                f"{selection.spec.name} did not prove configured model levers; "
                "require applied='configured', non-empty levers_applied, and no "
                "deferred levers"
            )
    if selection.spec.backend == "chai1":
        if not applied_levers:
            raise ActivationNotActiveError(
                f"{selection.spec.name} returned no applied-lever evidence; "
                "the activation state is unqualified"
            )
        _require_chai_byte_pin(report, api)
    return normalized_state or "applied"


def _require_chai_byte_pin(report: Mapping[str, Any], api: ActivationAPI) -> None:
    pins = report.get("pins")
    chai_pin = pins.get("chai_lab") if isinstance(pins, Mapping) else None
    if not isinstance(chai_pin, Mapping) or chai_pin.get("pinned") is not True:
        raise ActivationNotActiveError(
            f"{api.module}.{api.function} did not prove the Chai byte pin: "
            "pins.chai_lab.pinned is not True. A vendored or otherwise patched "
            "chai_lab module is incompatible with the pinned-wheel activation claim."
        )


def reset_activation_state() -> None:
    """Discard only pre-apply bookkeeping. This is NOT a module undo contract.

    Active receipts and poisoned/in-flight fences survive reset. A real worker
    restart gets a fresh interpreter; tests must isolate their fake registries.
    """
    with _ACTIVATION_LOCK:
        for module, state in list(_PROCESS_STATES.items()):
            if state == "PREPARING":
                del _PROCESS_STATES[module]


def _selection_key(selection: KitSelection) -> tuple[Any, ...]:
    return (
        selection.spec.name,
        selection.spec.backend,
        selection.model_identity,
        selection.mode,
        selection.revision,
        tuple(sorted(selection.env.items())),
        selection.activation_api.kind,
        selection.activation_api.module,
        selection.activation_api.function,
    )


def _validate_spec(name: str, spec: Mapping[str, Any]) -> None:
    ids = spec.get("softwareIds")
    modes = spec.get("modes")
    role = spec.get("role")
    install_args = spec.get("installArgs", [])
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"Anthropic kit {name} has invalid softwareIds")
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"Anthropic kit {name} has invalid modes")
    unknown = set(modes) - VALID_MODES
    if unknown:
        raise ValueError(
            f"Anthropic kit {name} has unknown mode(s): {', '.join(sorted(unknown))}"
        )
    if role not in {"replacement", "model_upgrade"}:
        raise ValueError(f"Anthropic kit {name} has invalid role {role!r}")
    if not isinstance(install_args, list) or not all(
        isinstance(value, str) for value in install_args
    ):
        raise ValueError(f"Anthropic kit {name} has invalid installArgs")


def _validate_kit_tree(root: Path, kit: str) -> None:
    kit_dir = root / kit
    for relative in ("run.sh", "opt/pyproject.toml"):
        path = kit_dir / relative
        if not path.is_file():
            raise SourceUnavailableError(
                f"Anthropic kit {kit} is incomplete: missing {path}"
            )


def _install_arg(spec: Mapping[str, Any], flag: str) -> str | None:
    args = [str(item) for item in spec.get("installArgs", ())]
    for index, item in enumerate(args[:-1]):
        if item == flag:
            return args[index + 1]
    return None


def _git_output(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        raise SourceUnavailableError(
            f"cannot inspect Anthropic kit source at {root}: {detail}"
        )
    return process.stdout.strip()

