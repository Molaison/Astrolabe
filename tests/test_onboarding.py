from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from astrolabe import onboarding


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrolabe")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    onboarding.register_commands(subparsers)
    return parser


def test_profiles_and_plans_are_immutable_catalog_views() -> None:
    assert onboarding.PROFILES["cpu"].software_ids == ()
    assert onboarding.PROFILES["protein"].software_ids == ("alphafold2", "ligandmpnn")
    assert onboarding.PROFILES["af3"].software_ids == ("alphafold3",)
    assert onboarding.PROFILES["rna"].software_ids == ("rhofold", "nampnn")
    assert len(onboarding.PROFILES["all"].software_ids) == len(onboarding.SOFTWARE)

    plan = onboarding.build_plan(profile="protein")
    with pytest.raises(FrozenInstanceError):
        plan.profile.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        onboarding.PROFILES["new"] = onboarding.PROFILES["cpu"]  # type: ignore[index]


def test_setup_plan_reports_catalog_assets_licenses_and_installer_argv(tmp_path: Path) -> None:
    installer = tmp_path / "scripts" / "install.py"
    installer.parent.mkdir()
    installer.write_text("# test installer\n", encoding="utf-8")
    weights = tmp_path / "af3"
    weights.mkdir()

    plan = onboarding.build_plan(
        profile="af3",
        root=tmp_path / "root",
        af3_weights=weights,
        accept_licenses=("esm3-noncommercial",),
        backend_args=("--activate-checkout",),
        installer_path=installer,
    )
    payload = onboarding.plan_payload(plan)

    assert (tmp_path / "root").exists() is False
    assert plan.installer_available is True
    assert str(weights) in plan.argv
    assert plan.argv[-1] == "--activate-checkout"
    assert plan.accepted_licenses == ("esm3-noncommercial",)
    assert plan.required_licenses == ()
    assert payload["software"][0]["name"] == "alphafold3"
    assert payload["software"][0]["asset_policy"] == "restricted"
    assert payload["software"][0]["asset_urls"]
    assert payload["software"][0]["license_id"] == "af3-terms"
    assert payload["argv"] == list(plan.argv)


def test_custom_profiles_reuse_catalog_resolution_and_reject_unknown_names() -> None:
    plan = onboarding.build_plan(
        profile="cpu",
        software=("rosetta", "rna"),
        accept_licenses=("rosetta-license", "pyrosetta-license"),
    )

    assert plan.profile.name == "custom"
    assert {"rosetta", "pyrosetta", "rhofold", "nampnn"} <= set(
        plan.profile.software_ids
    )
    assert plan.missing_licenses == ()

    with pytest.raises(onboarding.OnboardingError, match="Unknown software"):
        onboarding.build_plan(software=("not-a-software",))


def test_non_af3_plan_allows_missing_af3_and_activates_by_default() -> None:
    plan = onboarding.build_plan(profile="protein", dry_run=False)

    assert "--allow-missing-af3" in plan.argv
    assert "--activate-checkout" in plan.argv
    software_index = plan.argv.index("--software") + 1
    assert plan.argv[software_index] == "alphafold2,ligandmpnn"

    opt_out = onboarding.build_plan(
        profile="protein",
        dry_run=False,
        activate_checkout=False,
    )
    assert "--activate-checkout" not in opt_out.argv


def test_licenses_are_reported_and_never_accepted_automatically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = onboarding.build_plan(profile="all")

    assert "pyrosetta-license" in plan.required_licenses
    assert "ananas-academic" in plan.required_licenses
    assert "rosetta-license" in plan.required_licenses
    assert "esm3-noncommercial" in plan.required_licenses
    assert "esmfold2-aup" in plan.required_licenses
    assert plan.accepted_licenses == ()
    assert set(plan.required_licenses) == set(plan.missing_licenses)

    with pytest.raises(onboarding.OnboardingError, match="Unknown license"):
        onboarding.build_plan(accept_licenses=("not-a-license",))

    actual_plan = onboarding.build_plan(software=("pyrosetta",), dry_run=False)
    monkeypatch.setattr(
        onboarding.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("installer must not start before consent"),
    )
    with pytest.raises(onboarding.OnboardingError, match="pyrosetta-license"):
        onboarding.run_setup(actual_plan)


def test_af3_plan_requires_an_explicit_weights_path(tmp_path: Path) -> None:
    plan = onboarding.build_plan(profile="af3")
    assert plan.af3_weights is None
    assert any("--af3-weights" in warning for warning in plan.warnings)

    with pytest.raises(onboarding.OnboardingError, match="--af3-weights"):
        onboarding.run_setup(plan)

    missing = tmp_path / "missing"
    with pytest.raises(onboarding.OnboardingError, match="does not exist"):
        onboarding.build_plan(profile="af3", af3_weights=missing)


def test_actual_install_requires_source_installer_and_passes_argv_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "no-checkout" / "scripts" / "install.py"
    plan = onboarding.build_plan(
        software=("ligandmpnn",),
        dry_run=False,
        installer_path=missing,
    )
    with pytest.raises(onboarding.OnboardingError, match="source checkout"):
        onboarding.run_setup(plan)

    installer = tmp_path / "scripts" / "install.py"
    installer.parent.mkdir()
    installer.write_text("# test installer\n", encoding="utf-8")
    plan = onboarding.build_plan(
        software=("ligandmpnn",),
        dry_run=False,
        installer_path=installer,
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(onboarding.subprocess, "run", fake_run)
    assert onboarding.run_setup(plan) == 0
    assert calls[0][0] == list(plan.argv)
    assert calls[0][0][-1] == "--activate-checkout"
    assert "--allow-missing-af3" in calls[0][0]
    assert calls[0][0][calls[0][0].index("--software") + 1] == "ligandmpnn"
    assert calls[0][1]["check"] is False
    assert "shell" not in calls[0][1]


def test_register_commands_supports_plan_json_and_help(capsys) -> None:
    parser = _parser()
    namespace = parser.parse_args(
        ["setup", "--profile", "protein", "--dry-run"]
    )
    assert namespace._handler is onboarding._cmd_setup
    assert namespace._cuda_preinitialize is False
    assert namespace._handler(namespace) == 0
    text = capsys.readouterr().out
    assert "alphafold2" in text
    assert "ligandmpnn" in text
    assert "argv:" in text

    namespace = parser.parse_args(["setup", "--profile", "protein", "--json"])
    assert namespace._handler(namespace) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["profile"]["name"] == "protein"
    assert [item["name"] for item in payload["software"]] == [
        "alphafold2",
        "ligandmpnn",
    ]

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["demo", "--help"])
    assert exc.value.code == 0
    assert "--out-dir" in capsys.readouterr().out


def test_doctor_defaults_to_cpu_and_delegates_environment_scope(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    namespace = argparse.Namespace(
        env=None,
        task=None,
        resolve=False,
        show_ok=False,
        soft_fail=False,
        python_deps=False,
        strict=False,
        config_dir=None,
    )
    assert onboarding._cmd_doctor(namespace) == 0
    assert "CPU package: ok" in capsys.readouterr().out

    import astrolabe.cli as cli_module

    delegated: list[argparse.Namespace] = []

    def fake_environment_doctor(delegate_namespace):
        delegated.append(delegate_namespace)
        return 7

    monkeypatch.setattr(cli_module, "_cmd_doctor", fake_environment_doctor)
    namespace.task = "release_protein"
    assert onboarding._cmd_doctor(namespace) == 7
    assert delegated[0].task == "release_protein"
    assert delegated[0].format == "text"


def test_help_and_plan_do_not_import_torch_or_jax() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = r"""
import json
import sys
from astrolabe.onboarding import build_plan, plan_payload
plan = build_plan(profile="protein")
json.dumps(plan_payload(plan))
print(json.dumps({"torch": "torch" in sys.modules, "jax": "jax" in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "jax": False}


def test_cpu_demo_runs_real_geometry_reports_and_preserves_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cpu-demo"
    result = onboarding.run_cpu_demo(output)

    assert result["device"] == "CPU"
    assert result["model_inference"] is False
    assert result["network_required"] is False
    assert result["ca_contact_pair_count"] > 0
    assert (output / "raw/9tp_geometry/001_run/structures/step_000_ABC_model.cif").is_file()
    assert (output / "raw/9tp_geometry/001_run/scores.csv").is_file()
    assert (output / "scores.sqlite").is_file()
    assert (output / "selected").is_dir()
    assert (output / "report").is_dir()
    assert (output / "onboarding.json").is_file()

    provenance = json.loads(
        (output / "raw/9tp_geometry/001_run/provenance.json").read_text()
    )
    assert provenance["device"] == "CPU"
    assert "no inference" in provenance["analysis"]
    assert len(provenance["input_sha256"]) == 64

    cold_output = tmp_path / "cpu-demo-cold-process"
    code = f"""
import json
import sys
from astrolabe.onboarding import run_cpu_demo
run_cpu_demo({str(cold_output)!r})
print(json.dumps({{"torch": "torch" in sys.modules, "jax": "jax" in sys.modules}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "jax": False}
    assert (cold_output / "onboarding.json").is_file()

    with pytest.raises(onboarding.OnboardingError, match="preserve previous results"):
        onboarding.run_cpu_demo(output)
