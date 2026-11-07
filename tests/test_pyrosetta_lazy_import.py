"""Isolate optional native imports; never load or initialize real PyRosetta."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_PRELUDE = """
import builtins
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from types import ModuleType, SimpleNamespace

real_import = builtins.__import__
attempts = []
unavailable = ModuleNotFoundError("isolated optional dependency")

def block_pyrosetta(name, *args, **kwargs):
    if name == "pyrosetta" or name.startswith("pyrosetta."):
        attempts.append(name)
        raise unavailable
    return real_import(name, *args, **kwargs)

builtins.__import__ = block_pyrosetta
from astrolabe.losses import interaction
assert attempts == [], attempts

symbol_names = (
    "pyrosetta", "rosetta_hbonds", "PerResidueSasaMetric",
    "ContactMolecularSurfaceFilter", "HolesFilter",
    "ShapeComplementarityFilter", "MutateResidue",
)
assert all(getattr(interaction, name) is None for name in symbol_names)
assert not interaction._PYROSETTA_IMPORT_ATTEMPTED

def install_fake(*, before_last=None, failure=None):
    modules = {}
    for name in (
        "pyrosetta", "pyrosetta.rosetta", "pyrosetta.rosetta.core",
        "pyrosetta.rosetta.core.scoring",
        "pyrosetta.rosetta.core.simple_metrics",
        "pyrosetta.rosetta.core.simple_metrics.per_residue_metrics",
        "pyrosetta.rosetta.protocols",
        "pyrosetta.rosetta.protocols.simple_filters",
        "pyrosetta.rosetta.protocols.simple_moves",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    def forbidden_init(*args, **kwargs):
        raise AssertionError("loader must not initialize PyRosetta")

    class SasaMetric:
        def calculate(self, pose):
            return {1: 1.25, 2: 2.5}

    expected = dict(
        pyrosetta=modules["pyrosetta"],
        rosetta_hbonds=SimpleNamespace(),
        PerResidueSasaMetric=SasaMetric,
        ContactMolecularSurfaceFilter=type("ContactFilter", (), {}),
        HolesFilter=type("HolesFilter", (), {}),
        ShapeComplementarityFilter=type("ShapeFilter", (), {}),
        MutateResidue=type("MutateResidue", (), {}),
    )
    modules["pyrosetta"].init = forbidden_init
    modules["pyrosetta.rosetta.core.scoring"].hbonds = expected["rosetta_hbonds"]
    modules["pyrosetta.rosetta.core.simple_metrics.per_residue_metrics"].PerResidueSasaMetric = SasaMetric
    filters = modules["pyrosetta.rosetta.protocols.simple_filters"]
    for name in ("ContactMolecularSurfaceFilter", "HolesFilter", "ShapeComplementarityFilter"):
        setattr(filters, name, expected[name])
    modules["pyrosetta.rosetta.protocols.simple_moves"].MutateResidue = expected["MutateResidue"]

    def fake_import(name, *args, **kwargs):
        if name == "pyrosetta" or name.startswith("pyrosetta."):
            attempts.append(name)
            if name == "pyrosetta.rosetta.protocols.simple_moves":
                if before_last is not None:
                    before_last()
                if failure is not None:
                    raise failure
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    return expected
"""


def _run_isolated(code: str) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(REPO_ROOT), env.get("PYTHONPATH")) if path
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PRELUDE) + textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_runner_import_preserves_registry_without_importing_pyrosetta() -> None:
    _run_isolated("""
        import run
        from astrolabe.losses.base import loss_registry
        for name in (
            "ipae", "contact_molecular_surface", "chain_holes_diff",
            "buried_fraction_allala", "ligand_protein_hbond",
        ):
            assert loss_registry.get_function(name).__module__ == interaction.__name__
        for name in (
            "interface_shape_complementarity", "interface_hbond_count",
            "interface_polar_satisfaction", "short_range_interface_pair_energy",
        ):
            assert loss_registry.get_function(name).__module__ == "astrolabe.losses.rfd4_feedback"
        assert loss_registry.get_function("rfd4_native_ligand_reward").__module__ == "astrolabe.losses.rfd4_native"
        assert attempts == [], attempts
        assert "pyrosetta" not in sys.modules
        assert not interaction._PYROSETTA_IMPORT_ATTEMPTED
    """)


def test_first_use_publishes_symbols_without_initialization_or_reimport() -> None:
    _run_isolated("""
        expected = install_fake()
        assert interaction.sasa_calculator(object()).tolist() == [1.25, 2.5]
        for name, symbol in expected.items():
            assert getattr(interaction, name) is symbol, name
        first_attempts = list(attempts)
        assert first_attempts
        interaction._require_pyrosetta()
        assert attempts == first_attempts
        assert interaction._PYROSETTA_IMPORT_ATTEMPTED
        assert interaction._PYROSETTA_IMPORT_ERROR is None
    """)


@pytest.mark.parametrize("native_failure", [False, True])
def test_optional_import_failure_is_cached_with_original_cause(native_failure: bool) -> None:
    _run_isolated(f"""
        native_failure = {native_failure!r}
        cause = unavailable
        if native_failure:
            cause = RuntimeError("isolated native symbol load failure")
            install_fake(failure=cause)
        expected_message = (
            "PyRosetta is required for these losses/selectors but is not available. "
            "Install it (e.g. via `pyrosetta-installer`) or remove PyRosetta-based terms."
        )
        first_attempts = None
        for _ in range(2):
            try:
                interaction._require_pyrosetta()
            except ImportError as exc:
                assert str(exc) == expected_message
                assert exc.__cause__ is cause
            else:
                raise AssertionError("missing optional dependency must fail")
            if first_attempts is None:
                first_attempts = list(attempts)
            assert attempts == first_attempts
        assert interaction._PYROSETTA_IMPORT_ATTEMPTED
        assert interaction._PYROSETTA_IMPORT_ERROR is cause
        assert all(getattr(interaction, name) is None for name in symbol_names)
    """)


@pytest.mark.parametrize("fail", [False, True])
def test_concurrent_first_use_loads_once_and_publishes_only_complete_symbols(fail: bool) -> None:
    _run_isolated(f"""
        fail = {fail!r}
        entered = threading.Event()
        release = threading.Event()
        barrier = threading.Barrier(8)
        cause = RuntimeError("isolated native symbol load failure") if fail else None

        def before_last():
            entered.set()
            assert release.wait(10), "test did not release import"

        expected = install_fake(before_last=before_last, failure=cause)

        def require():
            barrier.wait(timeout=10)
            try:
                interaction._require_pyrosetta()
            except ImportError as exc:
                assert fail
                assert exc.__cause__ is cause
                return
            assert not fail
            assert all(getattr(interaction, name) is symbol for name, symbol in expected.items())

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(require) for _ in range(8)]
            try:
                assert entered.wait(10), "loader did not reach final symbol import"
                assert all(getattr(interaction, name) is None for name in symbol_names)
                assert not interaction._PYROSETTA_IMPORT_ATTEMPTED
            finally:
                release.set()
            for future in futures:
                future.result(timeout=10)

        assert attempts.count("pyrosetta") == 1
        assert all(count == 1 for count in Counter(attempts).values()), attempts
        assert interaction._PYROSETTA_IMPORT_ATTEMPTED
        if fail:
            assert all(getattr(interaction, name) is None for name in symbol_names)
        else:
            assert all(getattr(interaction, name) is symbol for name, symbol in expected.items())
    """)


def test_direct_symbol_helpers_guard_before_using_native_dependencies() -> None:
    _run_isolated("""
        class GuardReached(Exception):
            pass

        def guard():
            raise GuardReached

        interaction._require_pyrosetta = guard
        calls = (
            lambda: interaction._chain_region_selector(None, "A"),
            lambda: interaction._chain_group_region_selector(None, ["A", "B"]),
            lambda: interaction._per_residue_sasa_calculator(False),
            lambda: interaction._per_residue_sasa_calculator(True),
            lambda: interaction.calculate_holes_diff(),
            lambda: interaction._load_pose_cached("unused.pdb", None, ()),
            lambda: interaction.mutate_pose_to_all_ala(None),
        )
        for call in calls:
            try:
                call()
            except GuardReached:
                continue
            raise AssertionError("helper used native dependency without guard")
        assert attempts == []
    """)


def test_existing_global_and_guard_monkeypatch_patterns_remain_supported() -> None:
    _run_isolated("""
        # Existing load_pose tests replace only the module global.
        fake_module = SimpleNamespace()
        interaction.pyrosetta = fake_module
        interaction._require_pyrosetta()
        interaction.load_pose_with_optional_mmcif_pdb_fallback = (
            lambda module, *args, **kwargs: module
        )
        assert interaction._load_pose_cached("unused.pdb", None, ()) is fake_module
        assert not interaction._PYROSETTA_IMPORT_ATTEMPTED

        # Existing scoring tests replace the guard and individual native symbols.
        interaction.pyrosetta = None
        interaction._require_pyrosetta = lambda: None
        interaction.PerResidueSasaMetric = lambda: SimpleNamespace(
            calculate=lambda pose: {1: 3.0}
        )
        assert interaction.sasa_calculator(object()).tolist() == [3.0]
        assert attempts == []
    """)
