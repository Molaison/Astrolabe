from __future__ import annotations

import subprocess
import sys

import pytest

from astrolabe.mutators.wrappers.subprocess_protocol import read_message, stop_process


def test_partial_private_response_times_out_and_process_is_reaped():
    proc = subprocess.Popen([sys.executable, "-c", "import os,time;os.write(1,b'{');time.sleep(30)"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        with pytest.raises(TimeoutError):
            read_message(proc, bytearray(), timeout=0.1)
    finally:
        stop_process(proc)
    assert proc.poll() is not None


def test_private_response_is_bounded_before_json_allocation():
    proc = subprocess.Popen([sys.executable, "-c", "import os;os.write(1,b'x'*4096+b'\\n')"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        with pytest.raises(ValueError, match="byte budget"):
            read_message(proc, bytearray(), timeout=2, max_bytes=128)
    finally:
        stop_process(proc)
    assert proc.poll() is not None
