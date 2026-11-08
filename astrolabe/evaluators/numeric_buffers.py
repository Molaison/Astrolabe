"""Private, versioned numeric bundles with copy-on-write mmap array reads."""

import json
import mmap
import os
from pathlib import Path
import re
import stat
import struct


_MAX_MANIFEST = 16 * 1024 * 1024
_MAX_DATA = 1 << 40
_MAX_NODES = 100_000
_MAX_DEPTH = 64
_DTYPE = re.compile(r"[<>=|]?(?:b1|[iu][1248]|f[248]|c(?:8|16))\Z")


def _numeric_dtype(dtype):
    import numpy as np

    if dtype.hasobject or dtype.fields or dtype.subdtype or not _DTYPE.fullmatch(dtype.str):
        raise TypeError(f"unsupported numeric dtype: {dtype}")
    if dtype.kind == 'b' and dtype.itemsize != 1:
        raise TypeError(f"unsupported numeric dtype: {dtype}")
    return np.dtype(dtype)


def write_bundle(path, value):
    """Create a new private bundle. A failed publication remains caller-owned."""
    import numpy as np

    path = Path(path)
    os.mkdir(path, 0o700)
    chunks = []
    count = 0
    nodes = 0
    size = 0

    def add(raw):
        nonlocal size
        offset = size
        size += len(raw)
        if size > _MAX_DATA:
            raise ValueError('bundle data exceeds limit')
        chunks.append(raw)
        return {'offset': offset, 'length': len(raw)}

    def encode(item, depth=0):
        nonlocal count, nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError('bundle nesting or node limit exceeded')
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError('bundle dictionary keys must be strings')
            return {'kind': 'dict', 'items': [[key, encode(val, depth + 1)] for key, val in item.items()]}
        if type(item) in (list, tuple):
            return {'kind': type(item).__name__, 'items': [encode(val, depth + 1) for val in item]}
        if type(item) is np.ndarray:
            dtype = _numeric_dtype(item.dtype)
            if item.ndim > 32:
                raise ValueError('array has too many dimensions')
            if size + item.nbytes > _MAX_DATA:
                raise ValueError('bundle data exceeds limit')
            raw = memoryview(np.ascontiguousarray(item)).cast('B') if item.nbytes else memoryview(b'')
            count += 1
            return {'kind': 'array', 'dtype': dtype.str, 'shape': list(item.shape), **add(raw)}
        if isinstance(item, np.generic):
            dtype = _numeric_dtype(item.dtype)
            return {'kind': 'scalar', 'dtype': dtype.str, **add(item.tobytes())}
        if type(item) is bytes:
            return {'kind': 'bytes', **add(item)}
        if item is None:
            return {'kind': 'none'}
        if type(item) is bool:
            return {'kind': 'bool', 'value': item}
        if type(item) is int:
            return {'kind': 'int', 'value': str(item)}
        if type(item) is float:
            return {'kind': 'float', 'bits': struct.pack('>d', item).hex()}
        if type(item) is str:
            return {'kind': 'str', 'value': item}
        raise TypeError(f"unsupported bundle leaf: {type(item).__name__}")

    root = encode(value)
    manifest = json.dumps({'version': 1, 'data_file': 'data.bin', 'value': root},
                          ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(manifest) > _MAX_MANIFEST:
        raise ValueError('bundle manifest exceeds limit')
    with os.fdopen(os.open(path / 'data.bin', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
        for chunk in chunks:
            output.write(chunk)
        if count and size == 0:
            output.write(b'\0')
        output.flush()
    with os.fdopen(os.open(path / '.manifest.tmp', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
        output.write(manifest)
        output.flush()
    os.replace(path / '.manifest.tmp', path / 'manifest.json')
    return {'data_bytes': max(size, 1) if count else size,
            'manifest_bytes': len(manifest), 'array_count': count}


def _open_private(path, directory=False):
    before = os.lstat(path)
    expected = stat.S_IFDIR if directory else stat.S_IFREG
    if stat.S_IFMT(before.st_mode) != expected or before.st_uid != os.getuid():
        raise ValueError('bundle path must be a same-user regular file or directory')
    if before.st_mode & 0o077:
        raise ValueError('bundle path must be private')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    if directory:
        flags |= getattr(os, 'O_DIRECTORY', 0)
    fd = os.open(path, flags)
    after = os.fstat(fd)
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        os.close(fd)
        raise ValueError('bundle path changed during open')
    return fd


def read_bundle(path):
    """Read a private bundle. Array bases retain their copy-on-write mmap."""
    import numpy as np

    path = Path(path)
    directory = _open_private(path, directory=True)
    try:
        manifest_fd = _open_private(path / 'manifest.json')
        try:
            if os.fstat(manifest_fd).st_size > _MAX_MANIFEST:
                raise ValueError('bundle manifest exceeds limit')
            raw = os.read(manifest_fd, _MAX_MANIFEST + 1)
        finally:
            os.close(manifest_fd)
        data_fd = _open_private(path / 'data.bin')
    finally:
        os.close(directory)
    try:
        data_size = os.fstat(data_fd).st_size
        if data_size > _MAX_DATA:
            raise ValueError('bundle data exceeds limit')

        def unique_pairs(pairs):
            result = {}
            for key, val in pairs:
                if key in result:
                    raise ValueError('duplicate manifest key')
                result[key] = val
            return result

        manifest = json.loads(raw, object_pairs_hook=unique_pairs)
        if (type(manifest) is not dict or set(manifest) != {'version', 'data_file', 'value'}
                or type(manifest['version']) is not int or manifest['version'] != 1
                or manifest['data_file'] != 'data.bin'):
            raise ValueError('unsupported bundle manifest')
        spans = []
        nodes = 0
        empty_array = False

        def descriptor(node, expected):
            dtype_text = node.get('dtype')
            if type(dtype_text) is not str or not _DTYPE.fullmatch(dtype_text):
                raise ValueError('unsupported numeric dtype')
            dtype = _numeric_dtype(np.dtype(dtype_text))
            if dtype.str != dtype_text:
                raise ValueError('noncanonical numeric dtype')
            if expected == 'array':
                shape = node.get('shape')
                if (type(shape) is not list or len(shape) > 32
                        or any(type(dim) is not int or dim < 0 for dim in shape)):
                    raise ValueError('invalid array shape')
                elements = 1
                for dim in shape:
                    elements *= max(dim, 1)
                    if elements * dtype.itemsize > _MAX_DATA:
                        raise ValueError('array exceeds limit')
                length = 0 if 0 in shape else elements * dtype.itemsize
            else:
                shape = None
                length = dtype.itemsize
            return dtype, shape, length

        def span(node, length=None):
            offset, stored = node.get('offset'), node.get('length')
            if (type(offset) is not int or type(stored) is not int or offset < 0 or stored < 0
                    or (length is not None and stored != length) or offset + stored > data_size):
                raise ValueError('invalid bundle data span')
            if stored:
                spans.append((offset, offset + stored))
            return offset, stored

        def decode(node, depth=0):
            nonlocal nodes, empty_array
            nodes += 1
            if nodes > _MAX_NODES or depth > _MAX_DEPTH or type(node) is not dict:
                raise ValueError('invalid bundle node')
            kind = node.get('kind')
            fields = {
                'dict': {'kind', 'items'}, 'list': {'kind', 'items'},
                'tuple': {'kind', 'items'}, 'array': {'kind', 'dtype', 'shape', 'offset', 'length'},
                'scalar': {'kind', 'dtype', 'offset', 'length'},
                'bytes': {'kind', 'offset', 'length'}, 'none': {'kind'},
                'bool': {'kind', 'value'}, 'int': {'kind', 'value'},
                'float': {'kind', 'bits'}, 'str': {'kind', 'value'},
            }
            if kind not in fields or set(node) != fields[kind]:
                raise ValueError('invalid bundle node fields')
            if kind in ('list', 'tuple', 'dict'):
                items = node['items']
                if type(items) is not list:
                    raise ValueError('invalid bundle items')
                if kind == 'dict':
                    result = {}
                    for pair in items:
                        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
                            raise ValueError('invalid bundle dictionary entry')
                        result[pair[0]] = decode(pair[1], depth + 1)
                    return result
                values = [decode(item, depth + 1) for item in items]
                return values if kind == 'list' else tuple(values)
            if kind in ('array', 'scalar'):
                dtype, shape, length = descriptor(node, kind)
                offset, _ = span(node, length)
                if kind == 'array':
                    if mapping is None:
                        raise ValueError('array requires nonempty data file')
                    empty_array |= length == 0
                    return np.ndarray(shape, dtype=dtype, buffer=mapping, offset=offset)
                return np.frombuffer(mapping, dtype=dtype, count=1, offset=offset)[0]
            if kind == 'bytes':
                offset, length = span(node)
                return os.pread(data_fd, length, offset)
            if kind == 'none':
                return None
            if kind == 'bool' and type(node['value']) is bool:
                return node['value']
            if kind == 'str' and type(node['value']) is str:
                return node['value']
            if kind == 'int' and type(node['value']) is str and re.fullmatch(r'0|-?[1-9][0-9]*', node['value']):
                return int(node['value'])
            if kind == 'float' and type(node['bits']) is str and re.fullmatch(r'[0-9a-f]{16}', node['bits']):
                return struct.unpack('>d', bytes.fromhex(node['bits']))[0]
            raise ValueError('invalid bundle primitive')

        mapping = mmap.mmap(data_fd, 0, access=mmap.ACCESS_COPY) if data_size else None
        result = decode(manifest['value'])
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError('overlapping bundle data spans')
        if spans:
            if spans[0][0] != 0 or spans[-1][1] != data_size or any(
                    left[1] != right[0] for left, right in zip(spans, spans[1:])):
                raise ValueError('bundle data size does not match descriptors')
        elif data_size != (1 if empty_array else 0):
            raise ValueError('bundle data size does not match descriptors')
        return result
    finally:
        os.close(data_fd)
