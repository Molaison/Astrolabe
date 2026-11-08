from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import subprocess

import pytest

from astrolabe import startup_dependencies as dependencies


def _cfg(loss_type="interface_holes"):
    return {"loss": {"loss_terms": {"packing": {"_type": loss_type}}}}


def _script(tmp_path, body):
    # Only disposable shell fixtures are created executable; no chmod of any
    # bundled binary (or even of these fixtures) is needed.
    path = tmp_path / "DAlphaBall fixture"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    with os.fdopen(fd, "w") as handle:
        handle.write("#!/bin/sh\n" + body + "\n")
    return path


@pytest.fixture(autouse=True)
def _unset_override(monkeypatch):
    monkeypatch.delenv("GH_DALPHABALL", raising=False)


def _assert_error(cfg, path, reason, **kwargs):
    with pytest.raises(dependencies.StartupDependencyError) as error:
        dependencies.check_startup_dependencies(cfg, **kwargs)
    message = str(error.value)
    assert "loss.loss_terms.packing._type" in message
    assert str(path.absolute()) in message
    assert reason in message
    assert "Fix:" in message
    return message


@pytest.mark.parametrize("cfg", [
    {}, {"enable_pyrosetta": True}, {"structure_start": {"path": "input.pdb"}},
    _cfg("protomer_shape_complementarity"), _cfg("physical_interface_sc"),
    _cfg("bindcraft_interface_nres"), _cfg("bindcraft_interface_hydrophobicity"),
    {"loss": {"loss_terms": {"interface_holes": None}}},
    {"unrelated": {"_type": "interface_holes"}},
    {"stages": {"default": {"weighted_score_expression": "'interface_holes'"}}},
])
def test_unused_dependency_does_not_resolve_or_probe(cfg, monkeypatch):
    monkeypatch.setenv("GH_DALPHABALL", "/explicit/but/unused/missing")

    def forbidden(*args, **kwargs):
        pytest.fail("unused DAlphaBall must not be resolved, stat'ed or launched")

    monkeypatch.setattr(dependencies, "resolve_dalphaball_path", forbidden)
    monkeypatch.setattr(dependencies.subprocess, "run", forbidden)
    assert dependencies.check_startup_dependencies(cfg) is None


def test_all_defined_losses_are_computed_even_without_expression_reference():
    cfg = _cfg()
    cfg["stages"] = {"default": {"weighted_score_expression": "plddt"}}
    assert dependencies.dalphaball_required_by(cfg)


def test_stage_whitelist_and_disabled_stages():
    cfg = _cfg()
    cfg["stages"] = {"default": {"active_loss_terms": []}, "disabled": None}
    assert dependencies.dalphaball_required_by(cfg) == ()
    cfg["stages"]["later"] = {"active_loss_terms": ["packing"]}
    assert dependencies.dalphaball_required_by(cfg)


def test_stage_whitelist_does_not_inherit():
    cfg = _cfg()
    cfg["stages"] = {"default": {"active_loss_terms": []}, "later": {}}
    assert dependencies.dalphaball_required_by(cfg)


def test_implicit_loss_through_reachable_expression_terms():
    cfg = {
        "loss": {"expression_terms": {
            "quality": "-AB.interface_buried_unsatisfied_hbonds",
            "unused": "ligand_local_holes",
        }},
        "stages": {"default": {"weighted_score_expression": "quality"}},
    }
    assert dependencies.dalphaball_required_by(cfg) == (
        "loss.expression_terms.quality (interface_buried_unsatisfied_hbonds)",
    )


def test_expression_terms_cycles_are_bounded():
    cfg = {"loss": {"expression_terms": {"a": "b", "b": "a"}},
           "stages": {"default": {"weighted_score_expression": "a"}}}
    assert dependencies.dalphaball_required_by(cfg) == ()


def test_bundled_path_is_absolute_and_resolution_does_not_launch(monkeypatch):
    monkeypatch.setattr(dependencies.subprocess, "run", lambda *a, **k: pytest.fail("launch"))
    assert dependencies.resolve_dalphaball_path() == (
        Path(dependencies.__file__).resolve().parent / "assets" / "DAlphaBall.gcc"
    )


def test_override_expansion_anchors_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DALPHA_TEST_SUBDIR", "tools")
    monkeypatch.setenv("GH_DALPHABALL", "$DALPHA_TEST_SUBDIR/DAlphaBall")
    assert dependencies.resolve_dalphaball_path() == tmp_path / "tools" / "DAlphaBall"
    monkeypatch.setenv("GH_DALPHABALL", "~/DAlphaBall")
    assert dependencies.resolve_dalphaball_path() == Path.home() / "DAlphaBall"


@pytest.mark.parametrize("override", ["missing", "", "   "])
def test_invalid_explicit_override_never_falls_back(tmp_path, monkeypatch, override):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GH_DALPHABALL", override)
    path = tmp_path / override.strip() if override.strip() else tmp_path
    reason = "cannot stat" if override.strip() else "explicitly empty"
    _assert_error(_cfg(), path, reason)


def test_existing_non_executable_is_rejected_without_chmod(tmp_path, monkeypatch):
    path = tmp_path / "DAlphaBall"
    path.write_text("not executable\n")
    before = path.stat().st_mode
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    monkeypatch.setattr(dependencies.subprocess, "run", lambda *a, **k: pytest.fail("launch"))
    _assert_error(_cfg(), path, "execute permission denied")
    assert path.stat().st_mode == before


def test_directory_is_not_an_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_DALPHABALL", str(tmp_path))
    _assert_error(_cfg(), tmp_path, "not a regular file")


def test_existing_executable_with_invalid_format_is_rejected(tmp_path, monkeypatch):
    path = _script(tmp_path, "exit 0")
    path.write_text("not an ELF file or a script with a shebang\n")
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    _assert_error(_cfg(), path, "cannot execute/load file")


def test_existing_executable_with_missing_interpreter_is_rejected(tmp_path, monkeypatch):
    path = _script(tmp_path, "exit 0")
    path.write_text(f"#!{tmp_path}/missing-loader\n")
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    assert path.exists() and os.access(path, os.X_OK)
    _assert_error(_cfg(), path, "cannot execute/load file")


def test_execute_permission_race_is_reported(tmp_path, monkeypatch):
    path = _script(tmp_path, "exit 0")
    monkeypatch.setenv("GH_DALPHABALL", str(path))

    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(dependencies.subprocess, "run", denied)
    _assert_error(_cfg(), path, "Permission denied")


def test_dynamic_loader_diagnostic_is_preserved(tmp_path, monkeypatch):
    # Replay the real loader's failure signature via a subprocess fixture, not
    # a claim that this test host itself is missing libgfortran.
    path = _script(tmp_path, "printf '%s\\n' "
                   "'error while loading shared libraries: libgfortran.so.3: "
                   "cannot open shared object file: No such file or directory' >&2\nexit 127")
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    message = _assert_error(_cfg(), path, "libgfortran.so.3")
    assert "exited 127" in message
    assert "cannot open shared object file" in message


def test_probe_timeout_kills_and_reaps_executable(tmp_path, monkeypatch):
    # Builtins only: no sleep child holding inherited pipe descriptors open.
    path = _script(tmp_path, "while :; do :; done")
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    _assert_error(_cfg(), path, "timed out", timeout=0.1)


@pytest.mark.parametrize("output", ["", "usage: surf_vol", "1 nan 1", "1 1 inf",
                                        "1 0 1", "2 1 1", "1 1 1\\n2 1 1"])
def test_zero_exit_without_valid_protocol_response_is_not_success(tmp_path, monkeypatch, output):
    path = _script(tmp_path, f"printf '%b' '{output}'")
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    _assert_error(_cfg(), path, "invalid surf_vol one-sphere response")


def test_probe_uses_confirmed_protocol_timeout_and_unmodified_environment(tmp_path, monkeypatch):
    path = _script(tmp_path, "exit 0")
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    before = dict(os.environ)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, f"1 {4 * math.pi} {4 * math.pi / 3}\n", "")

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    assert dependencies.check_startup_dependencies(_cfg(), timeout=2.5) == path
    assert dict(os.environ) == before
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [str(path), "surf_vol"]
    assert kwargs["input"] == "NPOINTS\n1\nCOORDS\n0 0 0 1\nEND\n"
    assert kwargs["timeout"] == 2.5
    assert "env" not in kwargs and not kwargs.get("shell", False)


def test_valid_protocol_subprocess_fixture_succeeds(tmp_path, monkeypatch):
    body = '''[ "$#" = 1 ] && [ "$1" = surf_vol ] || exit 11
expected='NPOINTS
1
COORDS
0 0 0 1
END'
actual=$(cat)
[ "$actual" = "$expected" ] || exit 12
printf '1 12.566370614359172 4.188790204786391\\n'
'''
    path = _script(tmp_path, body)
    monkeypatch.setenv("GH_DALPHABALL", str(path))
    assert dependencies.check_startup_dependencies(_cfg()) == path


@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan])
def test_probe_timeout_must_be_bounded(timeout):
    _assert_error(_cfg(), dependencies.resolve_dalphaball_path(),
                  "timeout must be finite and positive", timeout=timeout)


def test_real_bundled_binary_protocol():
    # Verify the preflight against this runtime's actual loader result. The
    # complete SIF may lack libgfortran.so.3 even though the host can run it.
    path = dependencies.resolve_dalphaball_path()
    direct = subprocess.run([str(path), "surf_vol"], input=dependencies._PROBE_INPUT,
                            capture_output=True, text=True, timeout=5)
    if direct.returncode == 0:
        assert dependencies.check_startup_dependencies(_cfg()) == path
    else:
        with pytest.raises(dependencies.StartupDependencyError) as caught:
            dependencies.check_startup_dependencies(_cfg())
        assert f"exited {direct.returncode}" in str(caught.value)
        assert direct.stderr.strip() in str(caught.value)
