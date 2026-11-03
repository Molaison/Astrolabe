"""Conservative process isolation for the existing repeat lifecycle."""
import os
import pathlib
import subprocess
import sys
import tempfile
import socket
import time
import uuid
import importlib
import importlib.util
import hashlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from omegaconf import OmegaConf


def _child_env():
    env = os.environ.copy()
    roots = [str(pathlib.Path(p).resolve()) for p in sys.path if p]
    env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


class BrokerSession:
    def __init__(self, max_resident_models=1, *, max_resident_per_device=None, resource_limits=None):
        if type(max_resident_models) is not int or max_resident_models < 1:
            raise ValueError('max_resident_models must be a positive integer')
        self.max_resident_models = max_resident_models
        if max_resident_per_device is None:
            max_resident_per_device = max_resident_models
        if type(max_resident_per_device) is not int or max_resident_per_device < 1:
            raise ValueError('max_resident_per_device must be a positive integer')
        self.max_resident_per_device = max_resident_per_device
        from .contracts import ResourceLimits
        self.resource_limits = ResourceLimits.from_mapping(resource_limits).as_dict()

    def __enter__(self):
        if os.environ.get('ASTROLABE_CPU_OWNER') == '1' or os.environ.get('ASTROLABE_EXECUTION_ENDPOINT'):
            raise RuntimeError('nested execution broker sessions are not supported')
        from .admission import allocated_devices
        allocated_devices(os.environ.get('CUDA_VISIBLE_DEVICES') or '')
        self.directory = tempfile.TemporaryDirectory(prefix='astrolabe-session-')
        self.endpoint = '@astrolabe-' + uuid.uuid4().hex
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        read_fd, self.write_fd = os.pipe()
        token_read, token_write = os.pipe()
        self.env = _child_env()
        allocation = self.env.get('CUDA_VISIBLE_DEVICES') or ''
        # A broker must never accidentally connect to a parent broker.
        for key in ('ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN'):
            self.env.pop(key, None)
        self.env['ASTROLABE_EXECUTION_ARTIFACT_ROOT'] = self.directory.name
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'astrolabe.runtime.broker', self.endpoint,
                 str(token_read), str(read_fd), self.directory.name, str(self.max_resident_models),
                 str(self.max_resident_per_device), json.dumps(self.resource_limits)], env=self.env,
                pass_fds=(read_fd, token_read))
        except BaseException:
            os.close(read_fd)
            os.close(self.write_fd)
            os.close(token_read)
            os.close(token_write)
            self.directory.cleanup()
            raise
        os.close(read_fd)
        os.close(token_read)
        try:
            os.write(token_write, self.token.encode())
        except BaseException as exc:
            os.close(token_write)
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            os.close(token_write)
        try:
            deadline = time.monotonic() + 15
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError('execution broker exited during startup')
                try:
                    with socket.socket(socket.AF_UNIX) as probe:
                        probe.connect('\0' + self.endpoint[1:])
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('execution broker startup timed out')
                    time.sleep(.02)
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        self.env.update(ASTROLABE_EXECUTION_ENDPOINT=self.endpoint,
                        ASTROLABE_EXECUTION_TOKEN=self.token,
                        ASTROLABE_EXECUTION_ARTIFACT_ROOT=self.directory.name,
                        ASTROLABE_ALLOCATED_CUDA_DEVICES=allocation,
                        ASTROLABE_CPU_OWNER='1',
                        CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
                        HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                        JAX_PLATFORMS='cpu', GH_CUDA_PREINITIALIZE='0')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            self.env[name] = '1'
        return self

    def __exit__(self, exc_type, exc, traceback):
        from .client import OutcomeUnknown
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None
        # Broker holds in-flight leases until workers terminate; never remove
        # its artifact root while it can still be using request files.
        result = self.process.wait()
        retained = pathlib.Path(self.directory.name, 'retained.json').exists()
        if result or retained:
            # TemporaryDirectory's finalizer would otherwise remove files
            # while a guardian is still reaping an uncertain process tree.
            self.directory._finalizer.detach()
            if result and exc_type is None:
                raise OutcomeUnknown('broker cleanup was not confirmed; artifacts retained at ' + self.directory.name)
        else:
            self.directory.cleanup()


def validate_parallel(cfg):
    """Reject any component that has not explicitly opted into this path."""
    from .capabilities import (CPU_LOSSES, CPU_SELECTORS, SELECTOR_SOURCE_SHA256,
                               BROKER_EVALUATORS, CPU_EVALUATORS, declares_broker_protocol)
    from astrolabe.mutators.resident import BROKER_MUTATORS, CPU_MUTATORS
    from astrolabe.losses.base import extract_loss_terms, loss_registry
    from astrolabe.selectors.utils import extract_variables

    def plain(value):
        return OmegaConf.to_container(value, resolve=False) if OmegaConf.is_config(value) else value

    def inactive_hooks(value, known):
        value = plain(value)
        if value in (None, {}, [], 'null', 'None'):
            return True
        if not isinstance(value, dict):
            return False
        for name, config in value.items():
            if name not in known:
                return False
            if config not in (None, {}, 'null', 'None'):
                if not isinstance(config, dict) or config.get('enabled', False):
                    return False
        return True

    if plain(cfg.get('structure_start')) not in (None, {}, 'null', 'None'):
        raise ValueError('parallel runtime does not support structure_start')
    initializer = cfg.get('initializer') or {}
    if initializer.get('mode', 'random') not in ('random', 'all_a', 'aa_set', 'pssm', 'mask_mix', 'x_mix'):
        raise ValueError('parallel runtime requires a CPU-only initializer')
    if (cfg.get('predocker') or {}).get('predocker_terms'):
        raise ValueError('parallel runtime does not support predockers')
    if (cfg.get('per_residue_scores') or {}).get('enabled', False):
        raise ValueError('parallel runtime does not support per-residue plugins')
    if not inactive_hooks(cfg.get('pre_eval_hooks'), {'af3_antibody_msa_cache'}):
        raise ValueError('parallel runtime does not support evaluation hooks')
    if not inactive_hooks((cfg.get('evaluator') or {}).get('pre_eval_hooks'), {'af3_antibody_msa_cache'}):
        raise ValueError('parallel runtime does not support evaluation hooks')
    if not inactive_hooks(cfg.get('post_eval_hooks'), {'rosetta_minimize', 'rosetta_repack'}):
        raise ValueError('parallel runtime does not support evaluation hooks')
    stages = cfg.get('stages') or {}
    loss_cfg = cfg.get('loss') or {}
    loss_terms = loss_cfg.get('loss_terms') or {}
    registered = set(loss_registry.valid_losses)

    def check_loss(kind, alias):
        expected_module = CPU_LOSSES.get(kind)
        if expected_module is None:
            raise ValueError(f'parallel runtime does not support loss {alias}')
        if kind in registered:
            function = loss_registry.get_function(kind)
            if function.__module__ != expected_module or function.__name__ != kind:
                raise ValueError(f'parallel runtime cannot certify loss implementation {alias}')

    for alias, term in loss_terms.items():
        if term is not None:
            if not hasattr(term, 'get'):
                raise ValueError(f'parallel runtime cannot certify loss {alias}')
            check_loss(str(term.get('_type', '')), alias)

    # Param-free loss names are also selected directly from stage expressions.
    expression_terms = loss_cfg.get('expression_terms') or {}
    pending = [str(stage.get(key)) for stage in stages.values() if stage
               for key in ('weighted_score_expression', 'early_stopping',
                           'prefix_early_stopping', 'acceptance_threshold', 'acceptance_gate')
               if stage.get(key) not in (None, '', 'null', 'None')]
    referenced, seen_expr = set(), set()
    while pending:
        names = set(extract_loss_terms([pending.pop()]))
        referenced.update(names)
        for name in names:
            if name in expression_terms and name not in seen_expr:
                seen_expr.add(name)
                pending.append(str(expression_terms[name]))
    for name in (referenced & registered) - set(loss_terms):
        check_loss(name, name)
    evaluators = (cfg.get('evaluator') or {}).get('evaluator_terms') or {}
    mutators = (cfg.get('mutator') or {}).get('mutator_terms') or {}
    default_stage = stages.get('default') or {}
    active_mutators = set()
    for stage in stages.values():
        if not stage:
            continue
        for alias in stage.get('mutators') or default_stage.get('mutators') or []:
            term = mutators.get(alias) or {}
            kind = str(term.get('_type', '')).lower()
            if kind not in CPU_MUTATORS and kind not in BROKER_MUTATORS:
                raise ValueError(f'parallel runtime does not support mutator {alias}')
            if kind in BROKER_MUTATORS:
                execution = term.get('execution', (cfg.get('mutator') or {}).get('execution', 'auto'))
                if execution not in ('auto', 'daemon'):
                    raise ValueError(f'parallel runtime requires daemon execution for mutator {alias}')
            active_mutators.add(str(alias))
        aliases = [stage.get('evaluator') or default_stage.get('evaluator')]
        gate = stage.get('transition_consistency_gate') or {}
        if gate.get('evaluator'):
            aliases.append(gate['evaluator'])
        for alias in aliases:
            term = evaluators.get(alias) or {}
            typ = str(term.get('_type', '')).lower()
            if typ in CPU_EVALUATORS:
                continue
            if typ == 'alphafold3':
                if not inactive_hooks(term.get('pre_eval_hooks'), {'af3_antibody_msa_cache'}):
                    raise ValueError('parallel runtime does not support evaluation hooks')
                default_af3 = (cfg.get('evaluator') or {}).get('alphafold3', {})
                if not inactive_hooks(default_af3.get('pre_eval_hooks'), {'af3_antibody_msa_cache'}):
                    raise ValueError('parallel runtime does not support evaluation hooks')
            if typ in BROKER_EVALUATORS:
                name = 'mol1' if typ == 'mol-1' else typ
                default_backend = (cfg.get('evaluator') or {}).get(name, {}) or {}
                default_mode = 'daemon' if name == 'mol1' else 'legacy'
                mode = term.get('execution', default_backend.get('execution', default_mode))
                if mode == 'daemon' and declares_broker_protocol(BROKER_EVALUATORS[typ]):
                    continue
            raise ValueError(f'parallel runtime does not support evaluator {alias}')

    mutator_cfg = cfg.get('mutator') or {}
    expressions = mutator_cfg.get('selector_expression')
    selector_by_alias = {}
    if expressions is not None and ':' in str(expressions):
        for part in str(expressions).split(';'):
            name, *value = part.split(':')
            selector_by_alias[name.strip()] = ''.join(value).strip()
    elif expressions is not None:
        selector_by_alias = {alias: str(expressions) for alias in active_mutators}
    for alias in active_mutators:
        term = mutators.get(alias) or {}
        expression = term.get('position_selector')
        if expression is None:
            expression = selector_by_alias.get(alias, selector_by_alias.get('MCMC'))
        if expression in (None, '', 'None'):
            continue
        selected = extract_variables(str(expression))
        unknown = selected - set(CPU_SELECTORS)
        if unknown:
            raise ValueError(f'parallel runtime cannot certify selectors {sorted(unknown)}')
        for module in ({'utils'} | {CPU_SELECTORS[name][0] for name in selected}):
            spec = importlib.util.find_spec(f'astrolabe.selectors.{module}')
            if spec is None or not spec.origin:
                raise ValueError(f'parallel runtime cannot find selector implementation {module}')
            digest = hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest()
            if digest != SELECTOR_SOURCE_SHA256.get(module):
                raise ValueError(f'parallel runtime cannot certify selector implementation {module}')


def _next_limit(limit, maximum, *, pressure, slowed, metrics, previous):
    n = metrics['completed'] - previous['completed']
    queued = metrics['wait_seconds'] - previous['wait_seconds']
    service = metrics['service_seconds'] - previous['service_seconds']
    pressure |= bool(n and queued > service) or metrics['waiting'] >= max(1, limit)
    return max(1, limit - 1) if pressure or slowed else min(maximum, limit + 1)


def run_parallel(configs, max_active, stop_requested, on_completed, *, max_resident_models=1,
                 max_resident_per_device=None, resource_limits=None, control=None):
    """Overlap only independent existing trajectories; never create trial work."""
    from .admission import allocated_devices
    from .broker_client import BrokerClient
    from .client import WorkerError
    from .contracts import ResourceLimits
    from .controller import AdaptiveController, ControlPolicy
    from .telemetry import EventLog
    limits = ResourceLimits.from_mapping(resource_limits)
    policy = ControlPolicy.parse(control)
    maximum = min(max_active, limits.cpu_slots)
    if maximum < 1:
        raise ValueError('max_active must be positive')
    remaining = iter(configs)
    exhausted = False
    active = {}
    completed = 0
    trace = EventLog(os.environ.get('ASTROLABE_RUNTIME_TRACE'))
    controller = AdaptiveController(maximum, policy=control, started=time.monotonic(), events=trace)
    trajectory_events = deque(maxlen=256)
    with BrokerSession(max_resident_models, max_resident_per_device=max_resident_per_device,
                       resource_limits=limits.as_dict()) as session, ThreadPoolExecutor(max_workers=maximum) as pool:
        devices = allocated_devices(session.env['ASTROLABE_ALLOCATED_CUDA_DEVICES'])
        admitted = 0
        monitor = object.__new__(BrokerClient)
        monitor.endpoint, monitor.token, monitor.limit = session.endpoint, session.token, 1048576
        def observed_metrics():
            metrics = monitor._rpc({'op': 'metrics'}, 5, submitted=False)['metrics']
            metrics['effective_completed'] = completed
            metrics['effective_recent'] = list(trajectory_events)
            return metrics
        while not exhausted or active:
            while not exhausted and len(active) < controller.limit and not stop_requested():
                try:
                    cfg = next(remaining)
                except StopIteration:
                    exhausted = True
                    break
                env = session.env.copy()
                if devices:
                    env['ASTROLABE_PREFERRED_CUDA_DEVICE'] = devices[admitted % len(devices)]
                else:
                    env.pop('ASTROLABE_PREFERRED_CUDA_DEVICE', None)
                admitted += 1
                future = pool.submit(run_trajectory, cfg, env=env)
                active[future] = time.monotonic()
            if not active:
                break
            done, _ = wait(active, timeout=policy.poll_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                started = active.pop(future)
                try:
                    future.result()
                except BaseException:
                    # Preserve the real failure/OOM signal before propagating
                    # the failed scientific trajectory. No request is replayed
                    # and no extra candidate is created for control feedback.
                    try:
                        failure_metrics = observed_metrics()
                        failure_metrics['trajectory_failure'] = True
                    except (WorkerError, KeyError, TypeError):
                        failure_metrics = None
                    controller.observe(failure_metrics, now=time.monotonic(),
                                       independent_work=False, active=len(active))
                    trace.write('trajectory_failure', duration=time.monotonic() - started,
                                effective_completed=completed, controller_reason=controller.last_reason)
                    raise
                completed += 1
                on_completed(completed)
                trajectory_events.append({'time': time.monotonic(), 'backend': 'cpu_trajectory',
                                          'shape_key': 'fixed_configured_work', 'phase': 'trajectory',
                                          'cold_start': completed == 1, 'outcome': 'completed',
                                          'duration': time.monotonic() - started})
                trace.write('trajectory_complete', completed=completed,
                            duration=time.monotonic() - started)
            try:
                metrics = observed_metrics()
                if metrics.get('stopping'):
                    raise RuntimeError('execution broker stopped after a resource/owner failure')
                # A CPU-only workflow without model calls still has a real
                # fixed-task completion measure. It is kept separate from GPU
                # request cohorts and never used to claim GPU utilization.
                if not metrics.get('recent') and not devices:
                    metrics['recent'] = list(trajectory_events)
                controller.observe(metrics, now=time.monotonic(),
                                   independent_work=not exhausted and not stop_requested(), active=len(active))
            except (WorkerError, KeyError, TypeError):
                controller.observe(None, now=time.monotonic(), independent_work=not exhausted, active=len(active))
    return completed


def run_trajectory(cfg, *, env=None):
    from .broker_client import BrokerClient
    child_env = env or _child_env()
    with tempfile.TemporaryDirectory(prefix='astrolabe-trajectory-') as tmp:
        config_path = pathlib.Path(tmp) / 'config.yaml'
        OmegaConf.save(cfg, config_path, resolve=False)
        process = subprocess.Popen([sys.executable, '-m', 'astrolabe.runtime.trajectory', str(config_path)],
                                   env=child_env, start_new_session=True)
        try:
            if child_env.get('ASTROLABE_EXECUTION_ENDPOINT'):
                monitor = object.__new__(BrokerClient)
                monitor.endpoint = child_env['ASTROLABE_EXECUTION_ENDPOINT']
                monitor.token, monitor.limit = child_env['ASTROLABE_EXECUTION_TOKEN'], 1048576
                monitor._rpc({'op': 'owner', 'pid': process.pid}, 5, submitted=False)
            result = process.wait()
            if result:
                raise subprocess.CalledProcessError(result, process.args)
        except BaseException:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
