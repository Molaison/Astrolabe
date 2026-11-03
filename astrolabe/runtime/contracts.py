"""Small, framework-free contracts at the execution boundary.

Unknown residency and memory estimates are represented by ``None``. In
particular, a live JSONL server is not evidence that weights are loaded or on
the GPU. These contracts do not alter scientific payloads or RNG state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import re


def identity_key(identity):
    if not isinstance(identity, dict):
        raise TypeError('backend identity must be an object')
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def performance_identity(identity):
    """Comparable workload identity, NEVER a key for sharing a model object.

    Only explicitly known ownership nonces are removed from the top-level
    adapter settings. Actual init config, code, weights, environment and
    precision stay intact. Device class stays; its allocation ordinal does
    not make identical models on identical cards different workloads.
    """
    value = json.loads(json.dumps(identity, allow_nan=False))
    settings = value.get('model_config')
    if isinstance(settings, dict):
        instance = settings.get('instance')
        if isinstance(instance, str):
            # The mutator wrapper adds this exact suffix to private work
            # directories. Preserve the configured base and all other paths.
            for section, names in (('config', ('output_dir', 'out_folder', 'work_dir')),
                                   ('kwargs', ('output_root',))):
                section_value = settings.get(section)
                if isinstance(section_value, dict):
                    suffix = '/resident-' + instance
                    for name in names:
                        path = section_value.get(name)
                        if isinstance(path, str) and path.endswith(suffix):
                            section_value[name] = path[:-len(suffix)] + '/resident-<instance>'
        settings.pop('trajectory', None)
        settings.pop('instance', None)
    device = value.get('device')
    if isinstance(device, dict):
        environment = value.get('environment')
        effective = environment.get('effective_env') if isinstance(environment, dict) else None
        if isinstance(effective, dict) and effective.get('CUDA_VISIBLE_DEVICES') == device.get('visible'):
            effective['CUDA_VISIBLE_DEVICES'] = '<allocated-device>'
        for name in ('visible', 'id', 'uuid', 'index'):
            device.pop(name, None)
    return identity_key(value)


def public_identity(identity):
    """Persist exact version/config manifests without plaintext credentials.

    Compatibility still uses the original identity. Redacted values retain a
    digest so an audit can distinguish configurations without recovering the
    secret. Broker credentials do not belong in model identity in any case.
    """
    secret = re.compile(r'(^|_)(password|passwd|secret|credential|api_key|access_token|auth_token|'
                        r'authorization|cookie|execution_token|broker_token|hf_token)($|_)', re.I)
    def redact(value):
        if isinstance(value, dict):
            return {key: ({'redacted_sha256': identity_key({'value': item})}
                          if secret.search(str(key)) else redact(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value
    return redact(json.loads(json.dumps(identity, allow_nan=False)))


def byte_count(value, name, *, optional=False):
    if optional and value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer' + (' or null' if optional else ''))
    return value


def lifecycle_snapshot(value, *, pid=None, epoch=None, load_count=None):
    """Validate an adapter's report without manufacturing loading evidence."""
    result = dict(value() if callable(value) else (value or {}))
    for name in ('weights_loaded', 'gpu_resident', 'ready'):
        if result.get(name) is not None and type(result[name]) is not bool:
            raise ValueError(f'lifecycle.{name} must be bool or null')
    count = result.get('load_count', load_count)
    if count is not None:
        byte_count(count, 'lifecycle.load_count')
    result.setdefault('weights_loaded', None)
    result.setdefault('gpu_resident', None)
    result.setdefault('ready', True)
    result['load_count'] = count
    result['process_alive'] = True
    result['pid'] = os.getpid() if pid is None else pid
    if epoch is not None:
        result['worker_epoch'] = epoch
    if 'resident_bytes' in result:
        byte_count(result['resident_bytes'], 'lifecycle.resident_bytes', optional=True)
    # Reject tensors, non-finite measurements and mutable framework objects.
    return json.loads(json.dumps(result, allow_nan=False))


@dataclass(frozen=True)
class OperationResources:
    resident_bytes: int | None = None
    scratch_bytes: int | None = None
    prepared_bytes: int = 0
    output_bytes: int | None = None
    ram_bytes: int | None = None
    cpu_slots: int = 1

    @classmethod
    def parse(cls, value=None):
        value = dict(value or {})
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown execution resources: {sorted(unknown)}')
        result = cls(**value)
        for name in ('resident_bytes', 'scratch_bytes', 'output_bytes', 'ram_bytes'):
            byte_count(getattr(result, name), name, optional=True)
        for name in ('prepared_bytes', 'cpu_slots'):
            byte_count(getattr(result, name), name)
        if result.cpu_slots < 1:
            raise ValueError('cpu_slots must be positive; a pure device phase releases CPU admission explicitly')
        return result

    def as_dict(self):
        return asdict(self)


def request_metadata(value=None):
    result = dict(value or {})
    try:
        from astrolabe.execution_identity import current_execution_identity
    except ImportError:
        context = None
    else:
        context = current_execution_identity()
    if context is not None:
        result.setdefault('scientific_request_id', context['scientific_request_id'])
        result.setdefault('request_identity', context)
    result.setdefault('phase', 'composite')
    if result['phase'] not in ('device', 'cpu_prepare', 'cpu_finalize', 'composite', 'managed', 'probe'):
        raise ValueError('unknown execution phase')
    for field in ('shape_key', 'scientific_request_id'):
        if field in result and (not isinstance(result[field], str) or not result[field] or len(result[field]) > 4096):
            raise ValueError(f'{field} must be a bounded nonempty string')
    if 'cold' in result and type(result['cold']) is not bool:
        raise ValueError('cold must be bool')
    if 'request_identity' in result:
        identity_key(result['request_identity'])
    result['resources'] = OperationResources.parse(result.get('resources')).as_dict()
    return json.loads(json.dumps(result, allow_nan=False))


@dataclass(frozen=True)
class ResourceLimits:
    cpu_slots: int
    ram_bytes: int
    prepared_bytes: int
    output_bytes: int
    pending_items: int = 32
    vram_fraction: float = .90
    telemetry_interval: float = 1.0
    telemetry_timeout: float = 2.0
    request_output_bytes: int | None = None

    @classmethod
    def from_mapping(cls, value=None):
        from .admission import available_cpus
        from .telemetry import memory_allocation
        value = dict(value or {})
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown runtime resource limits: {sorted(unknown)}')
        cpus = available_cpus()
        memory = memory_allocation()['limit_bytes']
        # Bound queue memory independently of model/trajectory memory.
        defaults = dict(cpu_slots=cpus, ram_bytes=memory,
                        prepared_bytes=min(1024 ** 3, max(1, memory // 4)),
                        output_bytes=min(8 * 1024 ** 3, max(1, memory)))
        defaults.update(value)
        if defaults.get('request_output_bytes') is None:
            defaults['request_output_bytes'] = max(1, min(1024 ** 3, defaults['output_bytes'] // 2))
        result = cls(**defaults)
        for name in ('cpu_slots', 'ram_bytes', 'prepared_bytes', 'output_bytes', 'pending_items', 'request_output_bytes'):
            if type(getattr(result, name)) is not int or getattr(result, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if result.cpu_slots > cpus or result.ram_bytes > memory:
            raise ValueError('runtime CPU/RAM budget exceeds the current job allocation')
        if result.prepared_bytes > result.ram_bytes:
            raise ValueError('prepared byte budget exceeds RAM budget')
        if result.request_output_bytes > result.output_bytes:
            raise ValueError('per-request output cap exceeds job output budget')
        if (not isinstance(result.vram_fraction, (int, float)) or
                not math.isfinite(result.vram_fraction) or not 0 < result.vram_fraction <= 1):
            raise ValueError('vram_fraction must be in (0, 1]')
        for name in ('telemetry_interval', 'telemetry_timeout'):
            number = getattr(result, name)
            if not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError(f'{name} must be positive')
        return result

    def as_dict(self):
        return asdict(self)
