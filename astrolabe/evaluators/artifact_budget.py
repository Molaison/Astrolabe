"""Validate feature-tree budgets before tensor copies or archive allocation."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

DEFAULT_MAX_BYTES = 8 * 1024**3
DEFAULT_MAX_METADATA_BYTES = 8 * 1024**2
MAX_NODES = 100_000
MAX_DEPTH = 64
MAX_ARRAYS = 4096


def _budget(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def validate_dump(value, max_bytes: int, max_metadata_bytes: int) -> None:
    limit = _budget(max_bytes, 'max_bytes')
    metadata_limit = _budget(max_metadata_bytes, 'max_metadata_bytes')
    data_bytes = metadata_bytes = nodes = 0
    ancestors: set[int] = set()

    def walk(item, depth=0):
        nonlocal data_bytes, metadata_bytes, nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError('feature tree exceeds node/depth budget')
        torch = sys.modules.get('torch')
        if torch is not None and isinstance(item, torch.Tensor):
            data_bytes += item.numel() * item.element_size()
        elif isinstance(item, np.ndarray):
            data_bytes += item.nbytes
            if data_bytes > limit:
                raise ValueError('feature tree exceeds byte budget')
            if item.dtype.hasobject:
                for child in item.flat:
                    walk(child, depth + 1)
        elif isinstance(item, np.generic):
            walk(np.asarray(item), depth + 1)
        elif type(item) in (str, bytes):
            metadata_bytes += len(item.encode('utf-8') if isinstance(item, str) else item) * 2 + 64
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError('nonfinite feature metadata')
        elif type(item) in (dict, tuple, list):
            if id(item) in ancestors:
                raise ValueError('cyclic feature tree')
            ancestors.add(id(item))
            try:
                if isinstance(item, dict):
                    for key, child in item.items():
                        walk(key, depth + 1)
                        walk(child, depth + 1)
                else:
                    for child in item:
                        walk(child, depth + 1)
            finally:
                ancestors.remove(id(item))
        metadata_bytes += 64
        if data_bytes > limit or metadata_bytes > metadata_limit:
            raise ValueError('feature tree exceeds byte or metadata budget')

    walk(value)


def _shape_bytes(shape, dtype) -> int:
    if not isinstance(shape, (tuple, list)) or len(shape) > 32:
        raise ValueError('invalid feature array shape')
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError('invalid feature array dimension')
    return math.prod(shape) * np.dtype(dtype).itemsize


def validate_load(root: Path, max_bytes: int, max_metadata_bytes: int) -> dict:
    limit = _budget(max_bytes, 'max_bytes')
    metadata_limit = _budget(max_metadata_bytes, 'max_metadata_bytes')
    metadata_path, archive_path = root / 'tree.json', root / 'arrays.npz'
    if metadata_path.is_symlink() or archive_path.is_symlink():
        raise ValueError('feature artifacts must not be symbolic links')
    if metadata_path.stat().st_size > metadata_limit:
        raise ValueError('feature metadata exceeds byte budget')
    with metadata_path.open('rb') as handle:
        raw = handle.read(metadata_limit + 1)
    if len(raw) > metadata_limit:
        raise ValueError('feature metadata exceeds byte budget')
    def invalid_constant(value):
        raise ValueError(f'nonfinite feature metadata: {value}')
    metadata = json.loads(raw, parse_constant=invalid_constant)
    if not isinstance(metadata, dict) or metadata.get('version') != 1:
        raise ValueError('unsupported feature codec version')
    arrays = {}
    declared_bytes = nodes = 0
    def walk(node, depth=0):
        nonlocal declared_bytes, nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH or not isinstance(node, dict):
            raise ValueError('invalid or oversized feature tree')
        kind = node.get('t')
        if kind == 'array':
            key = node.get('key')
            if not isinstance(key, str) or not key.startswith('a') or not key[1:].isdigit() or key in arrays:
                raise ValueError('invalid or repeated feature array key')
            dtype = np.dtype(node['dtype'])
            if dtype.hasobject or dtype.fields is not None:
                raise ValueError('unsupported feature array dtype')
            size = _shape_bytes(node['shape'], dtype)
            if type(node.get('nbytes')) is not int or node['nbytes'] != size:
                raise ValueError('invalid declared feature bytes')
            declared_bytes += size
            arrays[key + '.npy'] = (tuple(node['shape']), dtype, size)
        elif kind == 'objects':
            size = _shape_bytes(node['shape'], np.dtype('O'))
            if len(node['v']) != math.prod(node['shape']):
                raise ValueError('invalid object feature shape')
            declared_bytes += size
            for child in node['v']:
                walk(child, depth + 1)
        elif kind in ('tuple', 'list'):
            for child in node['v']:
                walk(child, depth + 1)
        elif kind == 'dict':
            for pair in node['v']:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('invalid feature mapping')
                walk(pair[1], depth + 1)
        elif kind in ('numpy_scalar', 'torch'):
            walk(node['v'], depth + 1)
        elif kind == 'scalar':
            if type(node.get('v')) is float and not math.isfinite(node['v']):
                raise ValueError('nonfinite feature metadata')
        elif kind != 'bytes':
            raise ValueError('invalid feature node')
        if declared_bytes > limit or len(arrays) > MAX_ARRAYS:
            raise ValueError('feature arrays exceed byte/count budget')
    walk(metadata['tree'])
    if archive_path.stat().st_size > limit + MAX_ARRAYS * 16384:
        raise ValueError('feature archive exceeds byte budget')
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != len(arrays) or {e.filename for e in entries} != set(arrays):
            raise ValueError('unexpected or duplicate feature archive entries')
        for entry in entries:
            shape, dtype, size = arrays[entry.filename]
            if entry.file_size > size + 16384:
                raise ValueError('feature archive entry exceeds declared size')
            with archive.open(entry) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    actual_shape, _, actual_dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    actual_shape, _, actual_dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError('unsupported feature array format')
                if (actual_shape != shape or actual_dtype != dtype or
                        _shape_bytes(actual_shape, actual_dtype) != size or
                        entry.file_size != handle.tell() + size):
                    raise ValueError('feature archive header differs from declared metadata')
    return metadata
