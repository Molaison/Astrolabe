"""Inspect launcher argv with a fake Apptainer; never expose or initialize CUDA."""
import os
from pathlib import Path
import subprocess

import pytest

RUNTIME=Path('/case/config/001_binder_interface/01_astrolabe_beam/14_interface_autostart/02_runtime.sh')


@pytest.mark.parametrize('arguments,visible,gpu',[
    (['-m','astrolabe','refine','interface'], '2', True),
    (['-m','astrolabe.cli','refine','interface'], None, True),
    (['-m','astrolabe','--debug','refine','interface'], '2', True),
    (['-m','astrolabe','--config-dir','/config with spaces','refine','interface'], '2', True),
    (['-m','astrolabe','refine','interface','--prepare-only'], '2', False),
    (['-m','astrolabe','refine','interface','--service-socket','/rpc/owner.sock'], '2', False),
    (['-m','astrolabe','refine','interface','--service-socket=/rpc/owner.sock'], '2', False),
    (['-m','pytest'], '2', False),
    (['--gpu','-m','astrolabe.evaluators.af3_service'], '2', True),
])
def test_launcher_maps_gpu_only_when_native_task_needs_it(tmp_path,arguments,visible,gpu):
    fake=tmp_path/'apptainer'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    fake.chmod(0o755)
    script=tmp_path/'config/a/b/c/runtime.sh';script.parent.mkdir(parents=True)
    text=RUNTIME.read_text()
    original=next(line for line in text.splitlines() if line.startswith('APPTAINER='))
    script.write_text(text.replace(original,'APPTAINER='+str(fake)))
    capture=tmp_path/'args.txt';env=dict(os.environ,CAPTURE=str(capture),TMPDIR=str(tmp_path))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    if visible is not None:env['CUDA_VISIBLE_DEVICES']=visible
    subprocess.run(['bash',str(script),*arguments],env=env,check=True)
    args=capture.read_text().splitlines()
    assert ('--nv' in args)==gpu
    assert f'CUDA_VISIBLE_DEVICES={visible or "0"}' in args if gpu else 'CUDA_VISIBLE_DEVICES=' in args
    assert ('JAX_PLATFORMS=cpu' in args)==(not gpu)
