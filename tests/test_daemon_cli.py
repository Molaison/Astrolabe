from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

import pytest

from astrolabe import daemon_cli
from astrolabe.backend_catalog import BACKENDS
from astrolabe.model_daemon import MAX_FRAME_BYTES, PROTOCOL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrolabe")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    daemon_cli.register_commands(subparsers)
    return parser


def _write_dead_record(
    root: Path,
    name: str,
    *,
    lifecycle_status: str,
    socket_name: str = "model.sock",
    pid: int | None = None,
) -> Path:
    directory = root / name
    directory.mkdir(mode=0o700)
    identity = "a" * 64
    spec = {
        "protocol": PROTOCOL,
        "backend": "cpu-dead-test",
        "factory": "adapter:make",
        "settings": {},
        "identity": identity,
        "generation": "dead-generation",
        "max_pending": 2,
        "max_frame_bytes": MAX_FRAME_BYTES,
    }
    lifecycle = {
        "status": lifecycle_status,
        "pid": pid,
        "parent_pid": None,
        "socket": str(directory / socket_name),
        "identity": identity,
        "generation": "dead-generation",
        "exit_code": 0 if lifecycle_status in {"stopped", "exited"} else None,
        "message": f"last recorded state was {lifecycle_status}",
    }
    (directory / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (directory / "lifecycle.json").write_text(
        json.dumps(lifecycle),
        encoding="utf-8",
    )
    (directory / "spec.json").chmod(0o600)
    (directory / "lifecycle.json").chmod(0o600)
    return directory


def test_backend_listing_uses_catalog_without_model_frameworks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = r"""
import contextlib
import io
import json
import sys
from astrolabe.daemon_cli import main
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    code = main(['daemon', 'backends', '--json'])
payload = json.loads(buffer.getvalue())
print(json.dumps({
    'code': code,
    'count': len(payload),
    'names': [item['name'] for item in payload],
    'torch': 'torch' in sys.modules,
    'jax': 'jax' in sys.modules,
    'numpy': 'numpy' in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    payload = json.loads(completed.stdout)
    assert payload["code"] == 0
    assert payload["count"] == len(BACKENDS)
    assert {"alphafold3", "boltz", "ligandmpnn"} <= set(payload["names"])
    assert payload["torch"] is False
    assert payload["jax"] is False
    assert payload["numpy"] is False


def test_status_discovery_is_owner_only_and_never_calls_stale_records_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_cli.tempfile, "gettempdir", lambda: str(tmp_path))
    stopped = _write_dead_record(
        tmp_path,
        "astrolabe-model-daemon-stopped",
        lifecycle_status="stopped",
        pid=os.getpid(),
    )
    stale = _write_dead_record(
        tmp_path,
        "astrolabe-model-daemon-stale",
        lifecycle_status="ready",
    )
    unrelated = tmp_path / "some-other-daemon"
    unrelated.mkdir(mode=0o700)
    unsafe = _write_dead_record(
        tmp_path,
        "astrolabe-model-daemon-unsafe",
        lifecycle_status="stopped",
    )
    unsafe.chmod(0o755)

    records = daemon_cli.collect_status()
    by_name = {Path(item["directory"]).name: item for item in records}

    assert set(by_name) == {stopped.name, stale.name}
    assert by_name[stopped.name]["state"] == "stopped"
    assert by_name[stopped.name]["actual_rpc"] is False
    assert by_name[stopped.name]["ready"] is False
    assert by_name[stopped.name]["last_evidence"]["status"] == "stopped"
    assert by_name[stale.name]["state"] == "unknown"
    assert by_name[stale.name]["ready"] is False

    with pytest.raises(daemon_cli.DaemonCliError, match="0700"):
        daemon_cli.collect_status(directory=unsafe)


def test_stop_never_signals_a_pid_or_deletes_stale_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_cli.tempfile, "gettempdir", lambda: str(tmp_path))
    directory = _write_dead_record(
        tmp_path,
        "astrolabe-model-daemon-dead-pid",
        lifecycle_status="ready",
        pid=os.getpid(),
    )
    monkeypatch.setattr(
        os,
        "kill",
        lambda *_args, **_kwargs: pytest.fail("daemon CLI must never signal a recorded PID"),
    )

    with pytest.raises(daemon_cli.DaemonCliError, match="socket is not available"):
        daemon_cli.stop_daemon(directory=directory)
    assert (directory / "spec.json").is_file()
    assert (directory / "lifecycle.json").is_file()


def test_register_commands_sets_safe_handlers_and_parser_surface(capsys) -> None:
    parser = _parser()
    namespace = parser.parse_args(["daemon", "backends", "--json"])
    assert namespace._handler is daemon_cli._cmd_backends
    assert namespace._cuda_preinitialize is False
    assert namespace._handler(namespace) == 0
    listing = json.loads(capsys.readouterr().out)
    assert len(listing) == len(BACKENDS)

    namespace = parser.parse_args(["daemon", "status", "--json"])
    assert namespace._handler is daemon_cli._cmd_status
    assert namespace._cuda_preinitialize is False

    namespace = parser.parse_args(
        ["daemon", "stop", "--directory", "/tmp/astrolabe-model-daemon-example"]
    )
    assert namespace._handler is daemon_cli._cmd_stop
    assert namespace.directory == Path("/tmp/astrolabe-model-daemon-example")

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["daemon", "serve", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--spec" in help_text
    assert "--socket" not in help_text


def test_serve_reports_invalid_descriptor_without_traceback(
    tmp_path: Path, capsys
) -> None:
    code = daemon_cli.serve_foreground(tmp_path / "missing.json", poll_interval=0.01)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert code == 2
    assert events[0]["event"] == "service_failed"
    assert events[0]["error_type"] == "DaemonCliError"
    assert events[-1]["event"] == "service_exited"


def _write_cpu_adapter(tmp_path: Path) -> tuple[Path, Path]:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "cpu_adapter.py").write_text(
        "class Handler:\n"
        "    def __init__(self, settings):\n"
        "        self.settings = settings\n"
        "    def infer(self, payload):\n"
        "        return {'echo': payload.get('value')}\n"
        "\n"
        "def make(settings):\n"
        "    return Handler(settings)\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "service-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "backend": "cpu-daemon-cli-test",
                "factory": "cpu_adapter:make",
                "settings": {"variant": "cpu-test"},
                "python_bin": sys.executable,
                "cwd": str(Path(__file__).resolve().parents[1]),
                "max_pending": 2,
                "max_frame_bytes": MAX_FRAME_BYTES,
            }
        ),
        encoding="utf-8",
    )
    spec_path.chmod(0o600)
    return adapter_dir, spec_path


def _write_crashing_cpu_adapter(tmp_path: Path) -> tuple[Path, Path]:
    adapter_dir = tmp_path / "crashing-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "crashing_adapter.py").write_text(
        "import os\n"
        "import threading\n"
        "import time\n"
        "\n"
        "class Handler:\n"
        "    def infer(self, payload):\n"
        "        return payload\n"
        "\n"
        "def make(settings):\n"
        "    def exit_after_ready():\n"
        "        time.sleep(0.2)\n"
        "        os._exit(17)\n"
        "    threading.Thread(target=exit_after_ready, daemon=True).start()\n"
        "    return Handler()\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "crashing-service-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "backend": "cpu-daemon-crash-test",
                "factory": "crashing_adapter:make",
                "settings": {},
                "python_bin": sys.executable,
                "cwd": str(Path(__file__).resolve().parents[1]),
                "max_pending": 1,
                "max_frame_bytes": MAX_FRAME_BYTES,
            }
        ),
        encoding="utf-8",
    )
    spec_path.chmod(0o600)
    return adapter_dir, spec_path


@pytest.fixture
def service_temp_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="astrolabe-daemon-cli-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _service_env(adapter_dir: Path, temp_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(adapter_dir)
        if not existing
        else f"{adapter_dir}{os.pathsep}{existing}"
    )
    environment["TMPDIR"] = str(temp_root)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _read_event(master_fd: int, event_name: str, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    buffer = b""
    events: list[dict] = []
    while time.monotonic() < deadline:
        readable, _, _ = select.select([master_fd], [], [], 0.1)
        if not readable:
            continue
        try:
            chunk = os.read(master_fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            try:
                payload = json.loads(raw.strip())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            events.append(payload)
            if payload.get("event") == event_name:
                return payload
    raise AssertionError(
        f"timed out waiting for {event_name}: buffer={buffer!r}, events={events!r}"
    )


def _read_remaining_events(master_fd: int, timeout: float = 20) -> list[dict]:
    deadline = time.monotonic() + timeout
    buffer = b""
    events: list[dict] = []
    while time.monotonic() < deadline:
        readable, _, _ = select.select([master_fd], [], [], 0.1)
        if not readable:
            continue
        try:
            chunk = os.read(master_fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            try:
                events.append(json.loads(raw.strip()))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
    return events


def _start_service(spec_path: Path, environment: dict[str, str]):
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "astrolabe.daemon_cli",
            "daemon",
            "serve",
            "--spec",
            str(spec_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    ready = _read_event(master_fd, "service_ready")
    return process, master_fd, ready


def _run_daemon_cli(*arguments: str, environment: dict[str, str]) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "astrolabe.daemon_cli",
            "daemon",
            *arguments,
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def _stop_owned_service(process: subprocess.Popen, master_fd: int) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
    try:
        os.close(master_fd)
    except OSError:
        pass


def test_real_cpu_service_status_and_explicit_stop_through_cli_pty(
    tmp_path: Path, service_temp_root: Path
) -> None:
    adapter_dir, spec_path = _write_cpu_adapter(tmp_path)
    environment = _service_env(adapter_dir, service_temp_root)
    process, master_fd, ready = _start_service(spec_path, environment)
    try:
        directory = Path(ready["directory"])
        assert directory.name.startswith(daemon_cli.DAEMON_DIRECTORY_PREFIX)
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert directory.stat().st_uid == os.getuid()
        assert (directory / "spec.json").is_file()
        assert (directory / "lifecycle.json").is_file()
        assert (directory / "spec.json").stat().st_uid == os.getuid()
        assert (directory / "lifecycle.json").stat().st_uid == os.getuid()
        assert ready["status"]["state"] == "running"
        assert ready["status"]["model_load_count"] == 1

        status = _run_daemon_cli(
            "status",
            "--directory",
            str(directory),
            environment=environment,
        )
        record = status["daemons"][0]
        assert record["state"] == "running"
        assert record["ready"] is True
        assert record["actual_rpc"] is True
        assert record["generation_verified"] is False
        assert process.poll() is None

        stopped = _run_daemon_cli(
            "stop",
            "--directory",
            str(directory),
            environment=environment,
        )
        assert stopped["state"] == "drained"
        assert stopped["actual_rpc"] is True
        process.wait(timeout=10)
        assert process.returncode == 0
        events = _read_remaining_events(master_fd)
        by_name = {event["event"]: event for event in events}
        stopped_event = by_name["service_stopped"]
        exited_event = by_name["service_exited"]
        assert stopped_event["reason"] == "drained"
        assert exited_event["backend"] == "cpu-daemon-cli-test"
        assert directory.is_dir()
        lifecycle = json.loads((directory / "lifecycle.json").read_text())
        assert lifecycle["status"] == "stopped"
    finally:
        _stop_owned_service(process, master_fd)


def test_unexpected_child_exit_is_a_failure_with_retained_reason(
    tmp_path: Path, service_temp_root: Path
) -> None:
    adapter_dir, spec_path = _write_crashing_cpu_adapter(tmp_path)
    environment = _service_env(adapter_dir, service_temp_root)
    process, master_fd, ready = _start_service(spec_path, environment)
    try:
        assert ready["status"]["state"] == "running"
        process.wait(timeout=10)
        events = _read_remaining_events(master_fd)
        by_name = {event["event"]: event for event in events}
        stopped_event = by_name["service_stopped"]
        exited_event = by_name["service_exited"]
        assert process.returncode == 2
        assert stopped_event["reason"] == "child_failed"
        assert "code 17" in stopped_event["detail"]
        assert exited_event["backend"] == "cpu-daemon-crash-test"
    finally:
        _stop_owned_service(process, master_fd)


def test_sigterm_drains_foreground_owner_and_preserves_record(
    tmp_path: Path, service_temp_root: Path
) -> None:
    adapter_dir, spec_path = _write_cpu_adapter(tmp_path)
    environment = _service_env(adapter_dir, service_temp_root)
    process, master_fd, ready = _start_service(spec_path, environment)
    try:
        directory = Path(ready["directory"])
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        assert process.returncode == 0
        assert directory.is_dir()
        lifecycle = json.loads((directory / "lifecycle.json").read_text())
        assert lifecycle["status"] == "stopped"
        assert lifecycle["exit_code"] == 0
    finally:
        _stop_owned_service(process, master_fd)
