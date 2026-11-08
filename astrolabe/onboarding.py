"""Public installation planning, CPU diagnostics, and first-run demo.

The model installer remains the owner of source, asset, container, and GPU
setup. This module only builds a reviewable plan from the packaged software
catalog and invokes that installer as an argument list.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata, resources
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import Iterable, Sequence

from astrolabe import __version__, config_dir
from astrolabe.cli_command_definitions import set_handler
from astrolabe.software_catalog import (
    SOFTWARE,
    SOFTWARE_PROFILES,
    SoftwareSpec,
    resolve_software,
)


class OnboardingError(ValueError):
    """An onboarding request cannot proceed with the supplied inputs."""


@dataclass(frozen=True)
class InstallProfile:
    """A named, immutable software selection owned by the public onboarding CLI."""

    name: str
    summary: str
    software_ids: tuple[str, ...]


@dataclass(frozen=True)
class InstallPlan:
    """A complete, side-effect-free preview of one setup request."""

    profile: InstallProfile
    root: Path
    software: tuple[SoftwareSpec, ...]
    required_licenses: tuple[str, ...]
    accepted_licenses: tuple[str, ...]
    missing_licenses: tuple[str, ...]
    af3_weights: Path | None
    installer: Path
    installer_available: bool
    argv: tuple[str, ...]
    backend_args: tuple[str, ...]
    warnings: tuple[str, ...]
    dry_run: bool


_PROTEIN_SOFTWARE = resolve_software(("alphafold2", "ligandmpnn"))

PROFILES = MappingProxyType(
    {
        "cpu": InstallProfile(
            "cpu",
            "Packaged CPU CLI, configuration, diagnostics, and coordinate demo.",
            (),
        ),
        "protein": InstallProfile(
            "protein",
            "Lightweight public protein path using AF2 and LigandMPNN.",
            _PROTEIN_SOFTWARE,
        ),
        "af3": InstallProfile(
            "af3",
            "AlphaFold 3 with user-supplied licensed model parameters.",
            resolve_software(("alphafold3",)),
        ),
        "rna": InstallProfile(
            "rna",
            "The existing RNA catalog profile.",
            resolve_software(("rna",)),
        ),
        "all": InstallProfile(
            "all",
            "Every software ID in the installed catalog, with all preconditions.",
            resolve_software(("all",)),
        ),
    }
)

_AUTOMATIC_LICENSE_EXEMPTIONS = frozenset({"af3-terms"})
_VALID_LICENSE_IDS = frozenset(
    spec.license_id for spec in SOFTWARE.values() if spec.license_id is not None
)
_DEFAULT_ROOT = Path.home() / ".gh_envs"
_DEMO_RELATIVE_INPUT = Path("input") / "9TP_cid.cif"


def _split_repeated(values: Iterable[str] | None) -> tuple[str, ...]:
    result: list[str] = []
    for raw in values or ():
        for token in str(raw).split(","):
            value = token.strip().lower()
            if value and value not in result:
                result.append(value)
    return tuple(result)


def _resolve_software_values(values: Iterable[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    for value in _split_repeated(values):
        if value == "protein":
            expanded.extend(_PROTEIN_SOFTWARE)
        elif value in SOFTWARE_PROFILES:
            expanded.extend(resolve_software((value,)))
        elif value in SOFTWARE:
            expanded.append(value)
        else:
            choices = sorted(set(SOFTWARE) | set(SOFTWARE_PROFILES) | {"protein"})
            raise OnboardingError(
                f"Unknown software selection {value!r}; choose from: {', '.join(choices)}"
            )
    return resolve_software(expanded)


def _normalize_licenses(values: Iterable[str] | None) -> tuple[str, ...]:
    accepted = _split_repeated(values)
    unknown = sorted(set(accepted) - _VALID_LICENSE_IDS)
    if unknown:
        raise OnboardingError(
            "Unknown license ID(s): "
            + ", ".join(unknown)
            + ". Choose from: "
            + ", ".join(sorted(_VALID_LICENSE_IDS))
        )
    return accepted


def _resolved_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def build_plan(
    *,
    profile: str = "cpu",
    software: Iterable[str] | None = None,
    root: str | Path | None = None,
    af3_weights: str | Path | None = None,
    accept_licenses: Iterable[str] | None = None,
    dry_run: bool = True,
    backend_args: Sequence[str] = (),
    installer_path: str | Path | None = None,
    activate_checkout: bool = True,
) -> InstallPlan:
    """Build an installation plan without reading assets or writing files."""

    selected_override = _split_repeated(software)
    if selected_override:
        selected_ids = _resolve_software_values(selected_override)
        selected_profile = InstallProfile(
            "custom",
            "Software selection supplied explicitly on the command line.",
            selected_ids,
        )
    else:
        if profile not in PROFILES:
            raise OnboardingError(
                f"Unknown setup profile {profile!r}; choose from: {', '.join(PROFILES)}"
            )
        selected_profile = PROFILES[profile]
        selected_ids = selected_profile.software_ids

    specs = tuple(SOFTWARE[name] for name in selected_ids)
    required_licenses = tuple(
        sorted(
            {
                spec.license_id
                for spec in specs
                if spec.license_id is not None
                and spec.license_id not in _AUTOMATIC_LICENSE_EXEMPTIONS
            }
        )
    )
    accepted_licenses = _normalize_licenses(accept_licenses)
    missing_licenses = tuple(
        license_id
        for license_id in required_licenses
        if license_id not in accepted_licenses
    )

    root_path = (
        _resolved_path(root)
        if root is not None
        else _DEFAULT_ROOT.expanduser().resolve()
    )
    assert root_path is not None
    weights_path = _resolved_path(af3_weights)
    if weights_path is not None and not weights_path.is_dir():
        raise OnboardingError(f"AF3 weights directory does not exist: {weights_path}")

    default_installer = Path(__file__).resolve().parents[1] / "scripts" / "install.py"
    installer = _resolved_path(installer_path) or default_installer
    installer_available = installer.is_file()
    backend_values = tuple(str(value) for value in backend_args)

    warnings: list[str] = []
    if "protenix" in selected_ids:
        warnings.append(
            "Protenix is marked public-upstream-currently-broken in the software catalog."
        )
    if specs and not installer_available:
        warnings.append(
            "Actual model installation requires a source checkout containing "
            "scripts/install.py; the wheel contains plans and the CPU demo only."
        )
    if "alphafold3" in selected_ids and weights_path is None:
        warnings.append(
            "AlphaFold 3 installation requires --af3-weights pointing to authorized parameters."
        )
    if missing_licenses:
        warnings.append(
            "Actual installation requires explicit acceptance: "
            + ", ".join(f"--accept-license {item}" for item in missing_licenses)
        )

    argv: list[str] = []
    if specs:
        argv.extend(
            [
                sys.executable,
                str(installer),
                "--root",
                str(root_path),
                "--software",
                ",".join(selected_ids),
            ]
        )
        if weights_path is not None:
            argv.extend(["--af3-weights", str(weights_path)])
        if "alphafold3" not in selected_ids:
            argv.append("--allow-missing-af3")
        for license_id in accepted_licenses:
            argv.extend(["--accept-license", license_id])
        if activate_checkout:
            argv.append("--activate-checkout")
        argv.extend(backend_values)

    return InstallPlan(
        profile=selected_profile,
        root=root_path,
        software=specs,
        required_licenses=required_licenses,
        accepted_licenses=accepted_licenses,
        missing_licenses=missing_licenses,
        af3_weights=weights_path,
        installer=installer,
        installer_available=installer_available,
        argv=tuple(argv),
        backend_args=backend_values,
        warnings=tuple(warnings),
        dry_run=bool(dry_run),
    )


def plan_payload(plan: InstallPlan) -> dict[str, object]:
    """Return a JSON-safe representation with catalog-owned asset facts."""

    selected_assets = [
        {
            "name": spec.name,
            "summary": spec.summary,
            "asset_policy": spec.asset_policy,
            "asset_source": spec.asset_source,
            "asset_urls": list(spec.asset_urls),
            "revision": spec.revision,
            "license_id": spec.license_id,
            "requires": list(spec.requires),
            "host_envs": list(spec.host_envs),
            "submodules": list(spec.submodules),
        }
        for spec in plan.software
    ]
    return {
        "schema_version": 1,
        "dry_run": plan.dry_run,
        "profile": {
            "name": plan.profile.name,
            "summary": plan.profile.summary,
            "software_ids": list(plan.profile.software_ids),
        },
        "root": str(plan.root),
        "software": selected_assets,
        "licenses": {
            "required": list(plan.required_licenses),
            "accepted": list(plan.accepted_licenses),
            "missing": list(plan.missing_licenses),
        },
        "af3_weights": str(plan.af3_weights) if plan.af3_weights else None,
        "installer": str(plan.installer),
        "installer_available": plan.installer_available,
        "argv": list(plan.argv),
        "backend_args": list(plan.backend_args),
        "warnings": list(plan.warnings),
        "next_commands": [
            "astrolabe doctor",
            "astrolabe demo --out-dir outputs/cpu-demo",
        ],
    }


def _print_plan(plan: InstallPlan) -> None:
    payload = plan_payload(plan)
    print(f"profile: {plan.profile.name}")
    print(f"root: {plan.root}")
    print(f"installer: {plan.installer} ({'available' if plan.installer_available else 'missing'})")
    if plan.software:
        print("software:")
        for spec in plan.software:
            print(f"  {spec.name}: {spec.asset_policy}; {spec.asset_source}")
            if spec.license_id:
                print(f"    license: {spec.license_id}")
            for url in spec.asset_urls:
                print(f"    asset: {url}")
    else:
        print("software: none; the packaged CPU path needs no model download or container")
    print("licenses:")
    print(f"  required: {', '.join(plan.required_licenses) or 'none'}")
    print(f"  accepted: {', '.join(plan.accepted_licenses) or 'none'}")
    print(f"  missing: {', '.join(plan.missing_licenses) or 'none'}")
    print(f"af3_weights: {plan.af3_weights or 'not selected'}")
    print(f"argv: {json.dumps(payload['argv'])}")
    for warning in plan.warnings:
        print(f"warning: {warning}")
    print("next: astrolabe doctor")
    print("next: astrolabe demo --out-dir outputs/cpu-demo")


def run_setup(plan: InstallPlan) -> int:
    """Execute an install plan, or confirm the already installed CPU package."""

    if not plan.software:
        diagnostic = cpu_diagnostics()
        print(json.dumps(diagnostic, indent=2, sort_keys=True))
        if not diagnostic["cpu_ready"]:
            raise OnboardingError(
                "The packaged CPU path is incomplete. Reinstall Astrolabe from a built wheel."
            )
        return 0
    if plan.missing_licenses:
        raise OnboardingError(
            "Explicit license acceptance is required before installation: "
            + ", ".join(
                f"--accept-license {license_id}"
                for license_id in plan.missing_licenses
            )
        )
    if "alphafold3" in plan.profile.software_ids and plan.af3_weights is None:
        raise OnboardingError(
            "AlphaFold 3 setup requires --af3-weights pointing to authorized parameters."
        )
    if not plan.installer_available:
        raise OnboardingError(
            "Model installation requires a source checkout containing "
            f"{plan.installer}. Use the full astrolabe-source archive or Git checkout; "
            "the wheel supports CPU setup, planning, doctor, and demo only."
        )
    completed = subprocess.run(
        list(plan.argv),
        cwd=str(plan.installer.parent.parent),
        check=False,
    )
    return int(completed.returncode)


def _demo_input_bytes() -> bytes:
    checkout = Path(__file__).resolve().parents[1] / "examples" / _DEMO_RELATIVE_INPUT
    if checkout.is_file():
        return checkout.read_bytes()
    for package in ("examples.input", "examples"):
        try:
            resource = resources.files(package)
            candidate = (
                resource / "9TP_cid.cif"
                if package == "examples.input"
                else resource / _DEMO_RELATIVE_INPUT
            )
            if candidate.is_file():
                return candidate.read_bytes()
        except (ImportError, ModuleNotFoundError):
            continue
    raise FileNotFoundError(
        "Bundled demo input examples/input/9TP_cid.cif is missing. "
        "Reinstall Astrolabe from a current wheel or source checkout."
    )


def cpu_diagnostics() -> dict[str, object]:
    """Return checks that do not import model runtimes or inspect model assets."""

    reason = None
    try:
        _demo_input_bytes()
        demo_available = True
    except (FileNotFoundError, OSError) as exc:
        demo_available = False
        reason = str(exc)
    configs_available = (config_dir() / "base.yaml").is_file()
    installer = Path(__file__).resolve().parents[1] / "scripts" / "install.py"
    return {
        "astrolabe_version": __version__,
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "config_dir": str(config_dir()),
        "configs_available": configs_available,
        "demo_input_available": demo_available,
        "demo_input_error": reason,
        "source_installer_available": installer.is_file(),
        "cpu_ready": bool(sys.version_info >= (3, 11) and configs_available and demo_available),
        "next_commands": [
            "astrolabe demo --out-dir outputs/cpu-demo",
            "astrolabe setup --profile protein --dry-run",
        ],
    }


def _report_command(argv: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    install_root = str(Path(__file__).resolve().parents[1])
    python_path = os.pathsep.join(
        value
        for value in (install_root, os.environ.get("PYTHONPATH", ""))
        if value
    )
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": python_path,
        },
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(argv) + "\n")
        handle.write(completed.stdout)
        if completed.stderr:
            handle.write(completed.stderr)
        handle.write(f"\nexit_code={completed.returncode}\n")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise OnboardingError(
            f"Demo report command failed with exit code {completed.returncode}: {detail}"
        )


def run_cpu_demo(out_dir: str | Path) -> dict[str, object]:
    """Run coordinate geometry and report generation on the bundled AF3 input."""

    source_bytes = _demo_input_bytes()
    header = source_bytes[:4096].splitlines()
    if not any(line.strip().startswith(b"data_") for line in header):
        raise OnboardingError("Bundled demo input is not an mmCIF file")

    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise OnboardingError(
            f"Output already exists: {out}. Choose a new --out-dir to preserve previous results."
        )
    out.mkdir(parents=True, exist_ok=False)

    from omegaconf import OmegaConf

    from astrolabe.analysis.constraint_compliance import compute_geometry_metrics
    from astrolabe.core.binding_site import parse_binding_site

    specification = {
        "binder": "A:1-85",
        "target": "B:1-87",
        "mode": "contact",
        "max_distance": 8.0,
    }
    run_dir = out / "raw" / "9tp_geometry" / "001_run"
    structure_path = run_dir / "structures" / "step_000_ABC_model.cif"
    structure_path.parent.mkdir(parents=True)
    structure_path.write_bytes(source_bytes)

    spec = parse_binding_site(specification)
    if spec is None:
        raise OnboardingError("Demo binding-site specification did not parse")
    metrics = compute_geometry_metrics(structure_path, spec)
    metrics = {
        key: value
        for key, value in metrics.items()
        if "framework" not in key
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise OnboardingError("Demo geometry produced a non-finite value")

    OmegaConf.save(
        OmegaConf.create(
            {
                "task_name": "9tp_geometry",
                "oligo": ["ABC"],
                "analysis": {"kind": "coordinate_geometry", **specification},
            }
        ),
        run_dir / "config.yaml",
    )
    row = {"step": 0, "accepted": True, **metrics}
    with (run_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    def dependency_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unknown"

    provenance = {
        "analysis": "Astrolabe contact geometry on bundled AF3 prediction; no inference",
        "version": dependency_version("astrolabe"),
        "python": sys.version,
        "dependencies": {
            name: dependency_version(name)
            for name in ("numpy", "biopython", "omegaconf")
        },
        "input": "examples/input/9TP_cid.cif",
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "parameters": specification,
        "seed": None,
        "device": "CPU",
        "state": "geometry_completed",
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    log_path = out / "logs" / "posthoc.log"
    log_path.parent.mkdir(parents=True)
    cli = [sys.executable, "-m", "astrolabe.cli"]
    _report_command(
        [
            *cli,
            "posthoc",
            "collect",
            "--out-root",
            str(out / "raw"),
            "--db",
            str(out / "scores.sqlite"),
            "--export-csv",
            str(out / "scores.csv"),
            "--binder-chain",
            "A",
            "--target-chain",
            "B",
            "--workers",
            "1",
        ],
        cwd=out,
        log_path=log_path,
    )
    _report_command(
        [
            *cli,
            "posthoc",
            "filter",
            "--db",
            str(out / "scores.sqlite"),
            "--expr",
            "ca_allowed_min_dist <= 8",
            "--prefer-oligo",
            "ABC",
            "--dst",
            str(out / "selected"),
        ],
        cwd=out,
        log_path=log_path,
    )
    _report_command(
        [
            *cli,
            "posthoc",
            "report",
            "--db",
            str(out / "scores.sqlite"),
            "--prefer-oligo",
            "ABC",
            "--expr",
            "ca_allowed_min_dist <= 8",
            "--trend-metric",
            "ca_allowed_min_dist",
            "--dst",
            str(out / "report"),
        ],
        cwd=out,
        log_path=log_path,
    )

    result = {
        "output": str(out),
        "device": "CPU",
        "model_inference": False,
        "network_required": False,
        **metrics,
    }
    (out / "onboarding.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _backend_installer_args(namespace: argparse.Namespace) -> tuple[str, ...]:
    values: list[str] = []
    for flag, attribute in (
        ("--setup-host-envs", "setup_host_envs"),
        ("--fakeroot", "fakeroot"),
        ("--skip-build", "skip_build"),
        ("--skip-import-smoke", "skip_import_smoke"),
        ("--skip-doctor", "skip_doctor"),
        ("--no-nv", "no_nv"),
        ("--force", "force"),
    ):
        if bool(getattr(namespace, attribute, False)):
            values.append(flag)
    for flag, attribute in (
        ("--apptainer-bin", "apptainer_bin"),
        ("--sif", "sif"),
        ("--container-build-cache", "container_build_cache"),
    ):
        value = getattr(namespace, attribute, None)
        if value:
            values.extend([flag, str(value)])
    for flag, attribute in (
        ("--container-builder", "container_builder"),
        ("--runner-backend", "runner_backend"),
        ("--bootstrap-apptainer", "bootstrap_apptainer"),
    ):
        value = getattr(namespace, attribute, None)
        if value:
            values.extend([flag, str(value)])
    return tuple(values)


def _cmd_setup(namespace: argparse.Namespace) -> int:
    plan = build_plan(
        profile=namespace.profile,
        software=namespace.software,
        root=namespace.root,
        af3_weights=namespace.af3_weights,
        accept_licenses=namespace.accept_license,
        dry_run=bool(namespace.dry_run or namespace.json),
        backend_args=_backend_installer_args(namespace),
        activate_checkout=not namespace.no_activate_checkout,
    )
    if namespace.json:
        print(json.dumps(plan_payload(plan), indent=2, sort_keys=True))
        return 0
    if namespace.dry_run:
        _print_plan(plan)
        return 0
    return run_setup(plan)


def _cmd_doctor(namespace: argparse.Namespace) -> int:
    diagnostic = cpu_diagnostics()
    print(f"CPU package: {'ok' if diagnostic['cpu_ready'] else 'incomplete'}")
    print(f"version: {diagnostic['astrolabe_version']}")
    print(f"python: {diagnostic['python']}")
    print(f"configs: {diagnostic['config_dir']}")
    print(
        "bundled demo input: "
        + ("ok" if diagnostic["demo_input_available"] else str(diagnostic["demo_input_error"]))
    )
    print(
        "source model installer: "
        + ("available" if diagnostic["source_installer_available"] else "not packaged; source checkout required")
    )
    print("next: astrolabe demo --out-dir outputs/cpu-demo")
    print("next: astrolabe setup --profile protein --dry-run")

    delegate = any(
        (
            namespace.env,
            namespace.task,
            namespace.strict,
            namespace.python_deps,
        )
    )
    if not delegate:
        return 0 if diagnostic["cpu_ready"] else 2

    from astrolabe.cli import _cmd_doctor as environment_doctor

    env_namespace = argparse.Namespace(
        config_dir=getattr(namespace, "config_dir", None),
        env=namespace.env,
        task=namespace.task,
        resolve=namespace.resolve,
        format="text",
        show_ok=namespace.show_ok,
        soft_fail=namespace.soft_fail,
        python_deps=namespace.python_deps,
        strict=namespace.strict,
    )
    return int(environment_doctor(env_namespace))


def _cmd_demo(namespace: argparse.Namespace) -> int:
    result = run_cpu_demo(namespace.out_dir)
    print(json.dumps(result, indent=2, allow_nan=False, sort_keys=True))
    return 0


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the public onboarding commands on an existing CLI parser."""

    setup_parser = subparsers.add_parser(
        "setup",
        help="Plan or install a task profile.",
        description=(
            "Build an installation plan from the packaged software catalog or run "
            "the source installer. The default cpu profile downloads no assets."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    setup_parser.set_defaults(_cuda_preinitialize=False)
    setup_parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="cpu",
        help="Installation profile (default: cpu).",
    )
    setup_parser.add_argument(
        "--software",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help=(
            "Override the profile with catalog software, profiles, or 'protein'. "
            "Repeat or use comma-separated names."
        ),
    )
    setup_parser.add_argument(
        "--root",
        default=str(_DEFAULT_ROOT),
        help="Installer root (default: ~/.gh_envs).",
    )
    setup_parser.add_argument(
        "--af3-weights",
        help="Authorized AlphaFold 3 parameters directory; required for AF3 installation.",
    )
    setup_parser.add_argument(
        "--accept-license",
        action="append",
        default=[],
        metavar="ID[,ID...]",
        help="Explicitly accept a catalog license ID. Nothing is accepted automatically.",
    )
    setup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete plan without installing, downloading, or writing.",
    )
    setup_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete plan as JSON. JSON output is always plan-only.",
    )

    setup_parser.add_argument(
        "--no-activate-checkout",
        action="store_true",
        help=(
            "Build and verify the environment but do not point this source checkout "
            "at the generated runner."
        ),
    )
    setup_parser.add_argument("--setup-host-envs", action="store_true")
    setup_parser.add_argument("--fakeroot", action="store_true")
    setup_parser.add_argument("--skip-build", action="store_true")
    setup_parser.add_argument("--skip-import-smoke", action="store_true")
    setup_parser.add_argument("--skip-doctor", action="store_true")
    setup_parser.add_argument("--no-nv", action="store_true")
    setup_parser.add_argument("--force", action="store_true")
    setup_parser.add_argument("--apptainer-bin")
    setup_parser.add_argument("--sif", help="Existing SIF path; normally pair with --skip-build.")
    setup_parser.add_argument("--container-build-cache")
    setup_parser.add_argument(
        "--container-builder",
        choices=("apptainer", "docker"),
    )
    setup_parser.add_argument(
        "--runner-backend",
        choices=("apptainer", "docker-apptainer"),
    )
    setup_parser.add_argument(
        "--bootstrap-apptainer",
        choices=("auto", "always", "never"),
    )
    set_handler(setup_parser, _cmd_setup)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the CPU package and optionally environment paths.",
        description=(
            "With no arguments, check the packaged CPU path. Environment options "
            "are delegated lazily to the existing task-aware env doctor."
        ),
    )
    doctor_parser.set_defaults(_cuda_preinitialize=False)
    doctor_parser.add_argument("--env", help="Environment YAML for the existing env doctor.")
    doctor_parser.add_argument("--task", help="Task template for task-aware environment checks.")
    doctor_parser.add_argument("--resolve", action="store_true")
    doctor_parser.add_argument("--show-ok", action="store_true")
    doctor_parser.add_argument("--soft-fail", action="store_true")
    doctor_parser.add_argument("--python-deps", action="store_true")
    doctor_parser.add_argument("--strict", action="store_true")
    set_handler(doctor_parser, _cmd_doctor)

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the bundled CPU coordinate-analysis example.",
        description=(
            "Analyze bundled AF3 coordinates without model inference, network access, "
            "model weights, Torch, or JAX. Existing output directories are preserved."
        ),
    )
    demo_parser.set_defaults(_cuda_preinitialize=False)
    demo_parser.add_argument(
        "--out-dir",
        required=True,
        help="A new output directory; existing results are never overwritten.",
    )
    set_handler(demo_parser, _cmd_demo)


__all__ = [
    "InstallPlan",
    "InstallProfile",
    "OnboardingError",
    "PROFILES",
    "build_plan",
    "cpu_diagnostics",
    "plan_payload",
    "register_commands",
    "run_cpu_demo",
    "run_setup",
]
