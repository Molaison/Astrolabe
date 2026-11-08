"""Exercise the release entrypoint with real pytest in an isolated fixture checkout."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def release_runner(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/validation/release_cpu.py"
    spec = importlib.util.spec_from_file_location("release_cpu", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "test_contract.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "def test_selected():\n"
        f"    assert sys.executable == {sys.executable!r}\n"
        f"    assert Path.cwd() == Path({str(checkout)!r})\n"
        "    Path('executed').write_text('yes')\n"
        "def test_failure():\n"
        "    assert False, 'intentional fixture failure'\n"
    )
    monkeypatch.setattr(module, "REPO_ROOT", checkout)
    monkeypatch.setattr(module, "RELEASE_CPU_TESTS", ("test_contract.py",))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.chdir(tmp_path)
    return module, checkout


def test_release_runner_uses_checkout_interpreter_and_forwards_selection(release_runner):
    module, checkout = release_runner
    assert module.main(["-q", "-k", "selected", "-p", "no:cacheprovider"]) == 0
    assert (checkout / "executed").read_text() == "yes"


def test_release_runner_propagates_test_failure(release_runner):
    module, _ = release_runner
    assert module.main(["-q", "-k", "failure", "-p", "no:cacheprovider"]) == 1


def test_release_runner_forwards_collection_without_executing_tests(release_runner):
    module, checkout = release_runner
    assert module.main(["--collect-only", "-q", "-p", "no:cacheprovider"]) == 0
    assert not (checkout / "executed").exists()


def test_execution_suite_selects_its_tests_and_forwards_pytest_args(release_runner, monkeypatch):
    module, checkout = release_runner
    (checkout / "test_execution.py").write_text(
        "from pathlib import Path\n"
        "def test_execution_selected():\n"
        "    Path('execution-ran').write_text('yes')\n"
    )
    monkeypatch.setattr(module, "EXECUTION_CPU_TESTS", ("test_execution.py",))
    assert module.main(["--suite", "execution", "-q", "-k", "selected", "-p", "no:cacheprovider"]) == 0
    assert (checkout / "execution-ran").read_text() == "yes"
    assert not (checkout / "executed").exists()


def test_unknown_suite_fails_before_pytest(release_runner):
    module, checkout = release_runner
    with pytest.raises(SystemExit) as error:
        module.main(["--suite", "typo"])
    assert error.value.code == 2
    assert not (checkout / "executed").exists()
