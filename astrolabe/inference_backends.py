"""JAX completion bridge extracted from GH; no model runtime import at module load."""
from astrolabe.inference_pipeline import Pending

def jax_pending(value, *, wait_for, keepalive=None):
    """Advance only after the supplied JAX tree is ready, off the owner thread."""
    import jax
    return Pending(value, lambda: jax.block_until_ready(wait_for), keepalive)
