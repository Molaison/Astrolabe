"""Hold a GPU owner lock until its broker and registered process trees stop.

The parent-only control pipe closes even if the broker receives SIGKILL. The
guardian keeps the inherited flock while cleaning up; external subprocesses
need not inherit a lock descriptor to remain covered.
"""
import json
import os
import select
import signal
import sys
import time

from .telemetry import (ProcessObservationError, known_process_records,
                        process_identity, process_tree)


def supervise(control_fd, lock_fd):
    roots = {}
    known = {}
    groups = {}
    acknowledgements = {}
    last_error = None
    buffer = b''
    ended = False
    def observation_error(exc):
        nonlocal last_error
        message = str(exc)
        if message != last_error:
            print('GPU guardian retaining owner after unknown process observation: ' + message,
                  file=sys.stderr, flush=True)
            last_error = message

    def observe():
        identities = {**known, **roots}
        tree = process_tree(identities, groups=groups)
        tree.update(known_process_records(identities))
        known.update({pid: (pid, info['start']) for pid, info in tree.items()})
        return tree
    while not ended:
        ready, _, _ = select.select([control_fd], [], [], .1)
        if ready:
            raw = os.read(control_fd, 65536)
            if not raw:
                ended = True
            else:
                buffer += raw
                if len(buffer) > 65536:
                    raise ValueError('guardian control buffer exceeded')
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    record = json.loads(line)
                    identity = tuple(record['identity'])
                    # A registered root is known even before an enumeration
                    # contains it. Every use still verifies its start time.
                    roots[identity[0]] = known[identity[0]] = identity
                    if record.get('group'):
                        groups[identity[0]] = identity[1]
                    if record.get('ack_signal') == signal.SIGUSR1:
                        acknowledgements[identity[0]] = identity
        try:
            for pid, identity in list(acknowledgements.items()):
                if process_identity(pid) == identity:
                    os.kill(pid, signal.SIGUSR1)
                acknowledgements.pop(pid)
            tree = observe()
            present = {info['pgrp'] for info in tree.values()} | set(tree.unverified_groups)
            groups = {group: start for group, start in groups.items() if group in present}
        except (ProcessObservationError, OSError) as exc:
            # Keep groups and the lock while observation is uncertain. The
            # next pass retries; no missed snapshot is interpreted as exit.
            observation_error(exc)
    os.close(control_fd)
    # We keep checking every observed descendant, including children which
    # started their own session or whose original parent has already exited.
    term_sent = time.monotonic()
    while True:
        try:
            tree = observe()
            remaining = known_process_records({**known, **roots})
            if not remaining and not tree.unverified_groups:
                break
            sig = signal.SIGTERM if time.monotonic() - term_sent < 2 else signal.SIGKILL
            for pid, record in remaining.items():
                if process_identity(pid) != (pid, record['start']):
                    continue
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        except (ProcessObservationError, OSError) as exc:
            observation_error(exc)
        time.sleep(.05)
    os.close(lock_fd)


if __name__ == '__main__':
    supervise(int(sys.argv[1]), int(sys.argv[2]))
