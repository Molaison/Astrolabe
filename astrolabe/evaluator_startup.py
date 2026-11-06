"""Bounded construction of independent evaluator model owners."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
import hashlib
import importlib
import logging
import time


# These factories construct NativeModelClient owners; in-process construction
# and other wrappers retain their existing ordered initialization contract.
_INDEPENDENT_IMPORTS = {
    "alphafold2": ("astrolabe.evaluators.alphafold2", "astrolabe.evaluators.native_daemons"),
    "alphafold3": ("astrolabe.evaluators.alphafold3", "astrolabe.evaluators.wrappers.alphafold3",
                   "astrolabe.evaluators.native_daemons"),
    "esmfold": ("astrolabe.evaluators.esmfold", "astrolabe.evaluators.native_daemons"),
}

class EvaluatorInitializationError(RuntimeError):
    """A named evaluator failed during startup."""


def initialization_options(evaluator_conf):
    options = getattr(evaluator_conf, "initialization", None)
    if options is None and isinstance(evaluator_conf, dict):
        options = evaluator_conf.get("initialization")
    if options is None:
        return 1, None
    if not hasattr(options, "get"):
        raise ValueError("evaluator.initialization: expected a mapping")
    unknown = set(options) - {"workers", "seed"}
    if unknown:
        raise ValueError(f"evaluator.initialization: unknown keys {sorted(unknown)}")
    workers, seed = options.get("workers", 1), options.get("seed")
    if type(workers) is not int or workers < 1:
        raise ValueError("evaluator.initialization.workers must be a positive integer")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("evaluator.initialization.seed must be a uint32 integer or null")
    if workers > 1 and seed is None:
        raise ValueError(
            "Parallel evaluator initialization requires evaluator.initialization.seed; "
            "legacy initialization shares the caller's random stream."
        )
    return workers, seed


def validate_independent_plan(plan, seed):
    if seed is None:
        return
    from astrolabe.model_execution import execution_mode
    for alias, name, _factory, conf in plan:
        if name not in _INDEPENDENT_IMPORTS:
            raise ValueError(f"evaluator {alias!r} ({name}): independent initialization is unsupported")
        mode = conf.get("execution_mode", conf.get("execution"))
        if execution_mode(mode).value != "daemon" or conf.get("service_socket"):
            raise ValueError(f"evaluator {alias!r}: independent initialization requires a private native daemon")


def initialize_plan(plan, build, close, *, workers, seed):
    """Build independent owners, cancel native siblings on failure, retain order."""
    from astrolabe.model_execution import independent_initialization, model_construction_scope, startup_scope

    validate_independent_plan(plan, seed)
    # Imports may affect process-global state. Complete them before starting
    # threads; only native owner construction runs in the independent scopes.
    if seed is not None:
        for alias, name, _factory, conf in plan:
            try:
                if name == "alphafold2":
                    from astrolabe.evaluators.utils import _ensure_alphafold2_import_path
                    _ensure_alphafold2_import_path(conf.get("data_dir"))
                for module in _INDEPENDENT_IMPORTS[name]:
                    importlib.import_module(module)
            except BaseException as error:
                raise EvaluatorInitializationError(
                    f"Evaluator {alias!r} ({name}) import failed: {error}"
                ) from error
    completed = {}
    scope_context = startup_scope()
    scope = scope_context.__enter__()
    def construct(entry):
        alias, name, *_ = entry
        start = time.monotonic()
        logging.info("Initializing evaluator %s (%s)", alias, name)
        try:
            with scope.lock:
                if scope.cancelled:
                    raise RuntimeError("evaluator startup was cancelled")
            with model_construction_scope():
                if seed is None:
                    value = build(entry)
                else:
                    digest = hashlib.sha256(f"{seed}:{alias}".encode()).digest()
                    child_seed = int.from_bytes(digest[:4], "big")
                    with independent_initialization(child_seed):
                        value = build(entry)
        except BaseException as exc:
            reported = (EvaluatorInitializationError(
                f"Evaluator {alias!r} ({name}) initialization failed: {exc}"
            ) if isinstance(exc, Exception) else exc)
            with scope.lock:
                if scope.first_error is None:
                    scope.first_error = reported
            cleanup_errors = scope.cancel()
            for error in cleanup_errors:
                reported.add_note(f"startup cancellation cleanup failed: {error!r}")
            # Report immediately, before waiting for other constructors to drain.
            logging.error("Evaluator %s (%s) initialization failed after %.3fs: %s",
                          alias, name, time.monotonic() - start, exc)
            if not isinstance(exc, Exception):
                raise
            raise reported from exc
        logging.info("Evaluator %s (%s) ready in %.3fs", alias, name, time.monotonic() - start)
        return value

    failure = None
    futures = {}
    try:
        if workers == 1 or len(plan) < 2:
            for entry in plan:
                completed[entry[0]] = construct(entry)
        else:
            executor = ThreadPoolExecutor(max_workers=min(workers, len(plan)),
                                          thread_name_prefix="evaluator-startup")
            try:
                for entry in plan:
                    futures[executor.submit(copy_context().run, construct, entry)] = entry[0]
                for future in as_completed(futures):
                    try:
                        completed[futures[future]] = future.result()
                    except BaseException:
                        for pending in futures:
                            pending.cancel()
                        raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
    except BaseException as exc:
        exc = scope.first_error or exc
        failure = exc
        cleanup_errors = scope.cancel()
        for error in cleanup_errors:
            exc.add_note(f"startup cancellation cleanup failed: {error!r}")
        # Constructors already running may have completed during the drain.
        for future, alias in futures.items():
            if future.done() and not future.cancelled() and future.exception() is None:
                completed[alias] = future.result()
        for alias, value in reversed(list(completed.items())):
            try:
                close(value)
            except BaseException as error:
                logging.exception("Failed to release evaluator %s during startup cleanup", alias)
                exc.add_note(f"evaluator {alias!r} cleanup failed: {error!r}")
        raise exc
    finally:
        scope_context.__exit__(None, None, None)
        if failure is not None:
            logging.error("Evaluator startup aborted; completed owners have been released or cleanup errors reported")
    return {entry[0]: completed[entry[0]] for entry in plan}
