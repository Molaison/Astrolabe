"""Deterministic broker lanes and bounded JSONL framing (no client queue)."""
import hashlib
import json
import os
import socket
import time

BUSINESS_OPS = frozenset(('register', 'call', 'probe'))
CONTROL_OPS = frozenset(('cancel', 'metrics', 'detach', 'owner', 'stage', 'nested_process'))
NESTED_OPS = frozenset(('nested_begin', 'nested_end'))
LANE_OPS = {'business': BUSINESS_OPS, 'control': CONTROL_OPS, 'nested': NESTED_OPS}
MAX_CONTROL = 4 * 1024 * 1024
CONTROL_CAPACITY = 8
READ_TIMEOUT = 5.0
WRITE_TIMEOUT = 1.0


def lane_for(op):
    for lane, operations in LANE_OPS.items():
        if op in operations:
            return lane
    raise ValueError('unknown broker operation')


def endpoint_for(base, op):
    lane = lane_for(op)
    endpoint = base if lane == 'business' else base + '.' + lane
    # Linux sockaddr_un has 108 bytes, including the leading abstract NUL
    # or filesystem terminator. Long filesystem names use a private abstract
    # address, identically derived by all clients without extra environment.
    if len(os.fsencode(endpoint)) > 107:
        digest = hashlib.sha256(os.fsencode(base)).hexdigest()
        endpoint = '@astrolabe-' + digest + '.' + lane
    return endpoint


def unix_address(endpoint):
    return '\0' + endpoint[1:] if endpoint.startswith('@') else endpoint


def rejection(message):
    return {'ok': False, 'unknown': False, 'execution_state': 'not_executed',
            'retryable': True, 'error_type': 'AdmissionRejected', 'error': message}


def encode_response(result):
    return (json.dumps(result, allow_nan=False, separators=(',', ':')) + '\n').encode()


def read_line(sock, limit, deadline):
    """An absolute read deadline also bounds a producer sending one byte at a time."""
    chunks = bytearray()
    while not chunks.endswith(b'\n'):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError('broker socket read deadline expired')
        sock.settimeout(remaining)
        part = sock.recv(min(65536, limit + 1 - len(chunks)))
        if not part:
            raise EOFError('broker closed connection')
        chunks.extend(part)
        if len(chunks) > limit:
            raise ValueError('broker message too large')
        if b'\n' in chunks[:-1]:
            raise ValueError('multiple broker messages on one connection')
    return bytes(chunks)
