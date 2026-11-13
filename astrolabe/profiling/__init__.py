"""Opt-in, best-effort profiling without importing model libraries.

Span CPU time is that of the calling Python thread, not the sum of native
worker threads. No operation synchronizes an accelerator. cProfile is explicitly
opt-in and perturbs execution. Importing this package starts no collection.
"""

from .core import (
    ProfileConfig,
    child_environment,
    event,
    flush,
    profile_session,
    profiled,
    span,
)

__all__ = [
    "ProfileConfig", "profile_session", "span", "profiled", "event", "flush",
    "child_environment",
]
