"""Register with the resource guardian before executing a CUDA-capable command.

Invoked by absolute path with ``python -S``: no sitecustomize or model imports
may run before registration. exec preserves PID, process group and lock fd.
"""
import json
import os
import signal
import sys

control_fd = int(sys.argv[1])
command = json.loads(sys.argv[2])
with open('/proc/self/stat') as source:
    text = source.read()
pid = os.getpid()
if int(text.split(' ', 1)[0]) != pid:
    raise RuntimeError('process namespace cannot verify guarded launch')
fields = text[text.rfind(')') + 2:].split()
previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
record = {'identity': [pid, int(fields[19])], 'group': os.getpgrp() == pid,
          'ack_signal': signal.SIGUSR1}
os.write(control_fd, (json.dumps(record) + '\n').encode())
# The guardian must have recorded this exact process before model imports can
# run. A pipe write alone is insufficient if the leader exits before the
# guardian next gets CPU time. Blocking the acknowledgement signal first
# avoids a lost-signal race between registration and the wait.
if signal.sigtimedwait({signal.SIGUSR1}, 30) is None:
    raise RuntimeError('GPU resource guardian did not acknowledge guarded launch')
signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
os.close(control_fd)
os.execvpe(command[0], command, os.environ)
