from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


_DAEMON_FACTORY = "astrolabe.initializers.wrappers.rfd3_worker:daemon_factory"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_config(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        normalized = dict(config)
    else:
        normalized = OmegaConf.to_container(config, resolve=True)
    if not isinstance(normalized, dict):
        raise ValueError(
            f"RFD3 runner config must resolve to a mapping, got {type(normalized).__name__}."
        )
    return normalized


def _required_path(value: Any, *, field: str, executable: bool = False) -> Path:
    if value in (None, "", "null", "None"):
        raise ValueError(f"initializer.rfd3.{field} is required.")
    path = Path(os.path.abspath(Path(str(value)).expanduser()))
    if not path.exists():
        raise FileNotFoundError(f"initializer.rfd3.{field} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"initializer.rfd3.{field} is not executable: {path}")
    return path


@dataclass(frozen=True, slots=True)
class RFD3StructureOutput:
    example_id: str
    structure_path: Path
    metadata_path: Path | None


class RFD3Runner:
    """Run RFD3 requests in a persistent owned engine."""

    def __init__(self, config: Any, *, execution_mode: str | None = None):
        normalized = _normalize_config(config)
        configured_mode = normalized.pop("execution_mode", None)
        selected_mode = (
            execution_mode if execution_mode is not None else configured_mode
        )
        selected_mode = str(selected_mode or "daemon").strip().lower()
        if selected_mode not in {"daemon", "subprocess"}:
            raise ValueError("RFD3 execution_mode must be 'daemon' or 'subprocess'.")
        self.repo_path = _required_path(normalized.pop("repo_path", None), field="repo_path")
        self.python_bin = _required_path(
            normalized.pop("python_bin", None),
            field="python_bin",
            executable=True,
        )
        checkpoint = normalized.get("ckpt_path")
        if checkpoint in (None, "", "null", "None"):
            raise ValueError("initializer.rfd3.ckpt_path is required.")
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if checkpoint_path.is_absolute() or checkpoint_path.parent != Path("."):
            normalized["ckpt_path"] = str(
                _required_path(checkpoint_path, field="ckpt_path")
            )
        self.engine_config = normalized
        self.worker_path = Path(__file__).resolve().with_name("rfd3_worker.py")
        self.execution_mode = selected_mode
        self._daemon: Any = None
        self._daemon_settings: dict[str, Any] | None = None

    def _engine_settings(self) -> dict[str, Any]:
        return {
            "repo_path": str(self.repo_path),
            "engine_config": dict(self.engine_config),
        }

    def _worker_env(self) -> dict[str, str]:
        import_roots = [
            str(_repo_root()),
            str(self.repo_path / "src"),
            str(self.repo_path / "models" / "rfd3" / "src"),
        ]
        env = dict(os.environ)
        if env.get("PYTHONPATH"):
            import_roots.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(import_roots)
        return env

    def _daemon_spec(self, settings: dict[str, Any]) -> Any:
        from astrolabe.model_daemon import DaemonSpec

        return DaemonSpec(
            backend="rfd3",
            factory=_DAEMON_FACTORY,
            settings=settings,
            python_bin=str(self.python_bin),
            cwd=str(self.repo_path),
            env=self._worker_env(),
        )

    def _ensure_daemon(self, settings: dict[str, Any]) -> Any:
        if self._daemon is not None and self._daemon_settings == settings:
            return self._daemon
        self.shutdown()

        from astrolabe.model_daemon import OwnedModelDaemon

        self._daemon = OwnedModelDaemon(
            self._daemon_spec(settings),
            startup_timeout=300,
            request_timeout=None,
            shutdown_timeout=30,
            max_pending=4,
        )
        self._daemon_settings = settings
        return self._daemon

    @property
    def pid(self) -> int | None:
        return self._daemon.pid if self._daemon is not None else None

    def shutdown(self) -> None:
        daemon = self._daemon
        self._daemon = None
        self._daemon_settings = None
        if daemon is not None:
            daemon.shutdown()

    def _parse_response(self, response_path: Path) -> list[RFD3StructureOutput]:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        outputs = []
        for item in response.get("outputs") or []:
            structure_path = Path(str(item["structure_path"]))
            if not structure_path.is_file() or structure_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"RFD3 worker did not produce a non-empty structure: {structure_path}"
                )
            metadata_raw = item.get("metadata_path")
            metadata_path = Path(str(metadata_raw)) if metadata_raw else None
            outputs.append(
                RFD3StructureOutput(
                    example_id=str(item["example_id"]),
                    structure_path=structure_path,
                    metadata_path=metadata_path,
                )
            )
        if not outputs:
            raise RuntimeError("RFD3 returned no structure outputs.")
        return outputs

    def run(
        self,
        *,
        specification: dict[str, Any],
        output_dir: Path,
        n_batches: int,
    ) -> list[RFD3StructureOutput]:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "input.json"
        request_path = output_dir / "request.json"
        response_path = output_dir / "response.json"

        input_path.write_text(
            json.dumps({"gh_initializer": specification}, indent=2) + "\n",
            encoding="utf-8",
        )
        request_path.write_text(
            json.dumps(
                {
                    "repo_path": str(self.repo_path),
                    "config": self.engine_config,
                    "input_path": str(input_path),
                    "n_batches": int(n_batches),
                    "output_dir": str(output_dir),
                    "response_path": str(response_path),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        request = json.loads(request_path.read_text(encoding="utf-8"))
        if self.execution_mode == "subprocess":
            subprocess.run(
                [str(self.python_bin), "-u", str(self.worker_path), str(request_path)],
                cwd=str(self.repo_path),
                env=self._worker_env(),
                check=True,
            )
        else:
            self._ensure_daemon(self._engine_settings()).request(request)
        return self._parse_response(response_path)
