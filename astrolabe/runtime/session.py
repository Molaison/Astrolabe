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
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from omegaconf import OmegaConf


def _child_env():
    env = os.environ.copy()
    roots = [str(pathlib.Path(p).resolve()) for p in sys.path if p]
    env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


class BrokerSession:
    def __init__(self, max_resident_models=1, *, max_resident_per_device=None):
        if type(max_resident_models) is not int or max_resident_models < 1:
            raise ValueError('max_resident_models must be a positive integer')
        self.max_resident_models = max_resident_models
        if max_resident_per_device is None:
            max_resident_per_device = max_resident_models
        if type(max_resident_per_device) is not int or max_resident_per_device < 1:
            raise ValueError('max_resident_per_device must be a positive integer')
        self.max_resident_per_device = max_resident_per_device

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
                 str(self.max_resident_per_device)], env=self.env,
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
        except BaseException:
            os.close(token_write)
            self.__exit__(None, None, None)
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
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.env.update(ASTROLABE_EXECUTION_ENDPOINT=self.endpoint,
                        ASTROLABE_EXECUTION_TOKEN=self.token,
                        ASTROLABE_EXECUTION_ARTIFACT_ROOT=self.directory.name,
                        ASTROLABE_ALLOCATED_CUDA_DEVICES=allocation,
                        ASTROLABE_CPU_OWNER='1',
                        CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
                        HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                        JAX_PLATFORMS='cpu', GH_CUDA_PREINITIALIZE='0')
        return self

    def __exit__(self, *_):
        os.close(self.write_fd)
        # Broker holds in-flight leases until workers terminate; never remove
        # its artifact root while it can still be using request files.
        self.process.wait()
        self.directory.cleanup()


def validate_parallel(cfg):
    """Reject any component that has not explicitly opted into this path."""
    from .capabilities import CPU_LOSSES, CPU_SELECTORS, SELECTOR_SOURCE_SHA256
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
            if str(term.get('_type', '')).lower() != 'mcmc':
                raise ValueError(f'parallel runtime does not support mutator {alias}')
            active_mutators.add(str(alias))
        aliases = [stage.get('evaluator') or default_stage.get('evaluator')]
        gate = stage.get('transition_consistency_gate') or {}
        if gate.get('evaluator'):
            aliases.append(gate['evaluator'])
        for alias in aliases:
            term = evaluators.get(alias) or {}
            typ = str(term.get('_type', '')).lower()
            if typ == 'dummy':
                continue
            if typ == 'alphafold3':
                if not inactive_hooks(term.get('pre_eval_hooks'), {'af3_antibody_msa_cache'}):
                    raise ValueError('parallel runtime does not support evaluation hooks')
                default_af3 = (cfg.get('evaluator') or {}).get('alphafold3', {})
                if not inactive_hooks(default_af3.get('pre_eval_hooks'), {'af3_antibody_msa_cache'}):
                    raise ValueError('parallel runtime does not support evaluation hooks')
                mode = term.get('execution', default_af3.get('execution', 'legacy'))
                if mode == 'daemon':
                    try:
                        mod = importlib.import_module('astrolabe.evaluators.af3_resident')
                        if (getattr(mod, 'BROKER_SHARED_ROOT', False) is True and
                            getattr(mod, 'BROKER_ADMITTED_PROBE', False) is True):
                            continue
                    except ImportError:
                        pass
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
                 max_resident_per_device=None):
    """Adapt admission at trajectory completions using observed work and pressure."""
    from .admission import allocated_devices, available_cpus, resource_pressure
    from .broker_client import BrokerClient
    from .client import WorkerError
    remaining = iter(configs)
    exhausted = False
    limit = min(2, max_active)
    active = {}
    completed = 0
    previous_metrics = {'completed': 0, 'wait_seconds': 0.0, 'service_seconds': 0.0}
    duration_ema = None
    with BrokerSession(max_resident_models, max_resident_per_device=max_resident_per_device) as session, ThreadPoolExecutor(max_workers=max_active) as pool:
        devices = allocated_devices(session.env['ASTROLABE_ALLOCATED_CUDA_DEVICES'])
        admitted = 0
        monitor = object.__new__(BrokerClient)
        monitor.endpoint, monitor.token, monitor.limit = session.endpoint, session.token, 1048576
        while not exhausted or active:
            while not exhausted and len(active) < limit and not stop_requested():
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
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            durations = []
            for future in done:
                durations.append(time.monotonic() - active.pop(future))
                future.result()
                completed += 1
                on_completed(completed)
            observed = sum(durations) / len(durations)
            slowed = duration_ema is not None and observed > duration_ema * 1.5
            duration_ema = observed if duration_ema is None else .75 * duration_ema + .25 * observed
            pressure = resource_pressure(available_cpus())
            try:
                metrics = monitor._rpc({'op': 'metrics'}, 2, submitted=False)['metrics']
                limit = _next_limit(limit, max_active, pressure=pressure, slowed=slowed,
                                    metrics=metrics, previous=previous_metrics)
                previous_metrics = metrics
            except (WorkerError, KeyError, TypeError):
                limit = max(1, limit - 1)
    return completed


def run_trajectory(cfg, *, env=None):
    with tempfile.TemporaryDirectory(prefix='astrolabe-trajectory-') as tmp:
        config_path = pathlib.Path(tmp) / 'config.yaml'
        OmegaConf.save(cfg, config_path, resolve=False)
        subprocess.run([sys.executable, '-m', 'astrolabe.runtime.trajectory', str(config_path)],
                       check=True, env=env or _child_env())
