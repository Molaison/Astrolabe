"""Length-based AF3 padding, selected after native chemical tokenisation."""
from dataclasses import dataclass
import functools
import logging
import threading

PREFERRED_BUCKETS = (256, 512, 1024)
_INSTALL_LOCK = threading.Lock()


@dataclass(frozen=True)
class TokenBuckets:
    """A policy consumed by the native pipeline at its token-count boundary."""

    width: int = 40
    preferred: tuple[int, ...] = PREFERRED_BUCKETS
    tolerance: int = 40

    def select(self, num_tokens: int) -> int:
        for size in self.preferred:
            if size - self.tolerance <= num_tokens <= size:
                return size
        return ((num_tokens + self.width - 1) // self.width) * self.width


def install_bucket_dispatch(pipeline) -> None:
    """Extend the native selector; ordinary lists/None retain native semantics.

    Dispatch is stateless and installed once. Each request carries its immutable
    policy, so concurrent callers cannot overwrite one another's bucket choice.
    """
    with _INSTALL_LOCK:
        original = pipeline.calculate_bucket_size
        if getattr(original, "_astrolabe_token_buckets", False):
            return

        @functools.wraps(original)
        def calculate_bucket_size(num_tokens, buckets):
            if isinstance(buckets, TokenBuckets):
                size = buckets.select(num_tokens)
                logging.info("AF3 compilation bucket: tokens=%d padded=%d", num_tokens, size)
                return size
            return original(num_tokens, buckets)

        calculate_bucket_size._astrolabe_token_buckets = True
        pipeline.calculate_bucket_size = calculate_bucket_size
