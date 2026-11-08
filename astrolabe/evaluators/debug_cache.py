"""Optional initial-result cache for explicitly stateless evaluators.

This is not a checkpoint. Native owners without a replay contract, live poses,
and calls that consume global RNG are never reused through this shortcut.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import tempfile

from omegaconf import OmegaConf

from astrolabe.design.native_scores import snapshot_scores
from astrolabe.design.store import DataStore
from astrolabe.design.types import key_for
from astrolabe.model_execution import capture_random_state

logger = logging.getLogger(__name__)
_MAGIC = b"astrolabe.debug-initial/v1\n"


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


class InitialDebugCache:
    def __init__(self, path, binding, rng):
        self.path, self.binding, self.rng = Path(path), binding, rng

    @classmethod
    def prepare(cls, task_name, evaluator, args, inputs, runtime_config):
        contract = getattr(evaluator, "debug_cache_identity", None)
        if not callable(contract) or getattr(evaluator, "preserves_generated_coordinates", False):
            return None
        try:
            identity = contract()
            if identity is None:
                return None
            # Include executable implementation, not merely a class name or
            # the public package version (which can span multiple releases).
            package = Path(__file__).resolve().parents[1]
            files = set(package.rglob("*.py"))
            files.add(package.parent / "run.py")
            files.add(Path(inspect.getsourcefile(type(evaluator))))
            sources = tuple(
                (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in sorted(files)
            )
            rng = key_for(capture_random_state()).digest
            binding = key_for((
                "debug-raw-initial/v1", identity, sources, sys.version,
                inspect.getsource(evaluator.evaluate),
                _plain(args), inputs, _plain(runtime_config), rng,
            )).digest
            return cls(f"{task_name}_initial_debug.cache", binding, rng)
        except Exception as error:
            logger.info("Initial debug cache unavailable: %s", error)
            return None

    def load(self):
        if not self.path.exists():
            return None
        try:
            with self.path.open("rb") as stream:
                if stream.readline(128) != _MAGIC:
                    return None
                header = json.loads(stream.readline(4096))
                if header.get("binding") != self.binding or header.get("rng") != self.rng:
                    return None
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
                raise ValueError("debug cache payload checksum mismatch")
            result = pickle.loads(payload)
            if snapshot_scores(result, DataStore()).identity.digest != header["result_identity"]:
                raise ValueError("debug cache result or structure contents changed")
            return result
        except Exception as error:
            logger.warning("Ignoring invalid initial debug cache %s: %s", self.path, error)
            return None

    def publish(self, result):
        temporary = None
        try:
            if key_for(capture_random_state()).digest != self.rng:
                logger.info("Initial debug cache skipped: evaluation consumed RNG")
                return
            result_identity = snapshot_scores(result, DataStore()).identity.digest
            payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            header = json.dumps({
                "binding": self.binding, "rng": self.rng,
                "result_identity": result_identity,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }).encode() + b"\n"
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.path.parent,
                                             prefix=f".{self.path.name}.", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(_MAGIC + header + payload)
                stream.flush()
            os.replace(temporary, self.path)
        except Exception as error:
            # Cache publication must not turn a successful scientific call
            # into a failed/retriable model operation.
            logger.warning("Could not publish initial debug cache %s: %s", self.path, error)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning("Could not remove debug cache staging %s: %s", temporary, error)
