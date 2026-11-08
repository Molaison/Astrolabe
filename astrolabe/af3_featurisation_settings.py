"""Scope AF3 coordinator featurisation controls across evaluator aliases."""
from contextlib import contextmanager
import logging
import os
import threading

_LOCK = threading.RLock()
_KEYS = (
    "ALPHAFOLD3_REFSTRUCTURE_WORKERS",
    "ALPHAFOLD3_REFSTRUCTURE_CACHE",
    "ALPHAFOLD3_REFSTRUCTURE_CACHE_MAX_ENTRIES",
    "ALPHAFOLD3_MSA_CACHE",
    "ALPHAFOLD3_MSA_CACHE_MAX_ENTRIES",
)

def _configure_refstructure_env(config) -> None:
    """Apply RefStructure featurisation knobs via environment variables.

    These env vars are consumed inside the vendored AlphaFold3 codebase.
    Keeping them here allows Hydra/YAML config to control featurisation
    behaviour without requiring users to manually export env vars.
    """

    workers = getattr(config, "refstructure_workers", None)
    if workers is not None:
        try:
            os.environ["ALPHAFOLD3_REFSTRUCTURE_WORKERS"] = str(int(workers))
        except (TypeError, ValueError):
            logging.warning("Ignoring invalid refstructure_workers=%r", workers)

    cache = getattr(config, "refstructure_cache", None)
    if cache is not None:
        os.environ["ALPHAFOLD3_REFSTRUCTURE_CACHE"] = "1" if bool(cache) else "0"

    max_entries = getattr(config, "refstructure_cache_max_entries", None)
    if max_entries is not None:
        try:
            os.environ["ALPHAFOLD3_REFSTRUCTURE_CACHE_MAX_ENTRIES"] = str(
                int(max_entries)
            )
        except (TypeError, ValueError):
            logging.warning(
                "Ignoring invalid refstructure_cache_max_entries=%r", max_entries
            )


def _configure_msa_env(config) -> None:
    """Apply MSA featurisation cache knobs via environment variables."""

    cache = getattr(config, "msa_cache", None)
    if cache is not None:
        os.environ["ALPHAFOLD3_MSA_CACHE"] = "1" if bool(cache) else "0"

    max_entries = getattr(config, "msa_cache_max_entries", None)
    if max_entries is not None:
        try:
            os.environ["ALPHAFOLD3_MSA_CACHE_MAX_ENTRIES"] = str(int(max_entries))
        except (TypeError, ValueError):
            logging.warning("Ignoring invalid msa_cache_max_entries=%r", max_entries)



@contextmanager
def featurisation_environment(config):
    with _LOCK:
        previous = {key: os.environ.get(key) for key in _KEYS}
        try:
            if config is not None:
                _configure_refstructure_env(config)
                _configure_msa_env(config)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
