"""Configure JAX's native executable cache in the process owning the model."""
import logging
import os
from pathlib import Path
import tempfile

DEFAULT_CACHE_DIR = "~/.cache/jax_cache"


def configure_compilation_cache(config):
    import jax
    from jax.experimental.compilation_cache import compilation_cache

    directory = getattr(
        config, "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", DEFAULT_CACHE_DIR),
    )
    if not directory:
        compilation_cache.reset_cache()
        jax.config.update("jax_enable_compilation_cache", False)
        jax.config.update("jax_compilation_cache_dir", None)
        logging.info("AF3 persistent compilation cache disabled")
        return None

    path = Path(directory).expanduser().resolve()
    # A configured but unusable cache must not silently become a process-local
    # /tmp cache. Check writes too: mkdir(exist_ok=True) does not establish that.
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass
    # JAX 0.4 retains an initialized cache object and its enabled/disabled
    # decision separately from config. Reset those when changing either.
    if (jax.config.jax_compilation_cache_dir != str(path)
            or not jax.config.jax_enable_compilation_cache):
        compilation_cache.reset_cache()
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    logging.info("AF3 persistent compilation cache: %s", path)
    return path
