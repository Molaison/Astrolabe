"""Worker-side JSONL loop. The adapter loads and verifies its own model identity."""
import json
import os
import sys
import time
import uuid
import contextvars
from .contracts import lifecycle_snapshot, request_metadata, OperationResources

_CURRENT_REQUEST = contextvars.ContextVar('astrolabe_runtime_request', default=None)


def current_request():
    """Return control metadata for this handler, never a mutable scientific payload."""
    return dict(_CURRENT_REQUEST.get() or {})


def serve(handler, identity, *, load_count=None, max_message_bytes=1048576,
          lifecycle=None, resources=None, capabilities=None):
    if not isinstance(identity, dict):
        raise TypeError('identity must be an object')
    out = sys.stdout.buffer
    def send(obj):
        data = (json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if len(data) > max_message_bytes:
            raise ValueError('response exceeds max_message_bytes')
        out.write(data)
        out.flush()
    worker_epoch = uuid.uuid4().hex
    state = lifecycle_snapshot(lifecycle, epoch=worker_epoch, load_count=load_count)
    completed_requests = 0
    state['completed_requests'] = completed_requests
    send(dict(type='hello', protocol=1, identity=identity, pid=os.getpid(),
              load_count=state['load_count'], worker_epoch=worker_epoch,
              lifecycle=state, resources=OperationResources.parse(resources).as_dict(),
              capabilities=json.loads(json.dumps(capabilities or {}, allow_nan=False))))
    used = set()
    while True:
        line = sys.stdin.buffer.readline(max_message_bytes + 1)
        if not line:
            return
        if len(line) > max_message_bytes or not line.endswith(b'\n'):
            raise ValueError('request exceeds max_message_bytes')
        req = json.loads(line)
        if req.get('type') == 'shutdown':
            return
        rid, epoch = req.get('request_id'), req.get('epoch')
        if (req.get('type') != 'call' or not isinstance(rid, str) or not rid or
            not isinstance(epoch, str) or not epoch or not isinstance(req.get('payload'), dict) or
            (epoch, rid) in used):
            raise ValueError('invalid or duplicate request')
        if len(used) >= 100000:
            raise ValueError('worker request limit reached')
        used.add((epoch, rid))
        metadata = request_metadata(req.get('runtime'))
        started = time.monotonic()
        context_token = _CURRENT_REQUEST.set({**metadata, 'request_id': rid,
                                               'client_epoch': epoch, 'worker_epoch': worker_epoch})
        try:
            result = handler(req['payload'])
            if not isinstance(result, dict):
                raise TypeError('handler result must be an object')
            state = lifecycle_snapshot(lifecycle, epoch=worker_epoch, load_count=load_count)
            completed_requests += 1
            state['completed_requests'] = completed_requests
            send(dict(type='result', request_id=rid, epoch=epoch, result=result, lifecycle=state,
                      runtime={**metadata, 'worker_epoch': worker_epoch,
                               'handler_seconds': time.monotonic() - started}))
        except Exception as exc:
            send(dict(type='error', request_id=rid, epoch=epoch,
                      worker_epoch=worker_epoch, error_type=type(exc).__name__,
                      error=f'{type(exc).__name__}: {exc}'))
        finally:
            _CURRENT_REQUEST.reset(context_token)
