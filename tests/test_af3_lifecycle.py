"""Real subprocess/RPC lifecycle with a CPU echo service, never model inference."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf, open_dict

from astrolabe.evaluators import af3_lifecycle as lifecycle
from astrolabe.evaluators.inference_service import call
from astrolabe.protein_beam import load_config

TASK=Path(__file__).resolve().parents[1]/'config/interface.yaml'
INPUT=Path('/case/inputs/001_binder_interface/01_astrolabe_beam/03_bind_target_homologies')
FAKE='''
import os,sys
from astrolabe.evaluators.af3_lifecycle import guard_owner_parent
guard_owner_parent()
from astrolabe.evaluators.inference_service import serve,mark
mark('fake_model_load',cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
     jax=os.environ.get('JAX_PLATFORMS'),ld=os.environ.get('LD_LIBRARY_PATH'),
     torch='torch' in sys.modules,jax_imported='jax' in sys.modules)
serve(lambda payload:payload,sys.argv[1])
'''


@pytest.fixture
def config():
    cfg=load_config(TASK)
    with open_dict(cfg):
        cfg.protein_beam.service.startup_timeout_s=10
        cfg.protein_beam.service.shutdown_timeout_s=2
        cfg.protein_beam.service.terminate_timeout_s=1
    return cfg


@pytest.fixture
def fake_command(monkeypatch):
    monkeypatch.setattr(lifecycle,'_owner_command',lambda config,socket:[sys.executable,'-u','-c',FAKE,str(socket)])


def events(owner):
    return [json.loads(line) for line in owner.log_path.read_text().splitlines() if line.startswith('{')]


def test_service_settings_default_without_legacy_block_does_not_mutate_config():
    cfg = OmegaConf.create({'protein_beam': {}})
    before = OmegaConf.to_container(cfg, resolve=False)
    settings = lifecycle.service_settings(cfg)
    assert settings == lifecycle.AF3ServiceSettings(
        startup_timeout_s=600.0,
        shutdown_timeout_s=120.0,
        terminate_timeout_s=10.0,
        max_pending=4,
    )
    assert OmegaConf.to_container(cfg, resolve=False) == before
    assert 'service' not in cfg.protein_beam


def test_service_settings_partial_legacy_and_explicit_override():
    cfg = OmegaConf.create({
        'protein_beam': {
            'service': {
                'startup_timeout_s': 30,
                'max_pending': 2,
            }
        }
    })
    legacy = lifecycle.service_settings(cfg)
    assert legacy == lifecycle.AF3ServiceSettings(
        startup_timeout_s=30.0,
        shutdown_timeout_s=120.0,
        terminate_timeout_s=10.0,
        max_pending=2,
    )
    explicit = lifecycle.service_settings(
        cfg,
        explicit_settings={
            'shutdown_timeout_s': 7,
            'max_pending': 1,
        },
    )
    assert explicit == lifecycle.AF3ServiceSettings(
        startup_timeout_s=600.0,
        shutdown_timeout_s=7.0,
        terminate_timeout_s=10.0,
        max_pending=1,
    )


@pytest.mark.parametrize(
    'service',
    [
        {'unknown': 1},
        {'startup_timeout_s': 0},
        {'shutdown_timeout_s': float('nan')},
        {'terminate_timeout_s': float('inf')},
        {'startup_timeout_s': True},
        {'shutdown_timeout_s': '120'},
        {'max_pending': 0},
        {'max_pending': 1.5},
        {'max_pending': True},
    ],
)
def test_invalid_service_settings_fail_before_owner_directory(
    service,
    tmp_path,
):
    cfg = OmegaConf.create({
        'protein_beam': {'service': service},
        'evaluator': {'alphafold3': {}},
    })
    with pytest.raises((TypeError, ValueError)):
        lifecycle.OwnedAF3(cfg, tmp_path, os.environ)
    assert not (tmp_path / 'services').exists()


def test_owned_af3_accepts_explicit_settings_without_cfg_service_block(tmp_path):
    cfg = OmegaConf.create({
        'protein_beam': {},
        'evaluator': {'alphafold3': {}},
    })
    owner = lifecycle.OwnedAF3(
        cfg,
        tmp_path,
        os.environ,
        settings={'startup_timeout_s': 12, 'max_pending': 1},
    )
    assert owner.startup_timeout == 12.0
    assert owner.shutdown_timeout == 120.0
    assert owner.terminate_timeout == 10.0
    assert owner.max_pending == 1
    assert 'service' not in cfg.protein_beam


def test_owned_af3_uses_defaults_without_cfg_service_block(tmp_path):
    cfg = OmegaConf.create({
        'protein_beam': {},
        'evaluator': {'alphafold3': {}},
    })
    owner = lifecycle.OwnedAF3(cfg, tmp_path, os.environ)
    assert owner.startup_timeout == 600.0
    assert owner.shutdown_timeout == 120.0
    assert owner.terminate_timeout == 10.0
    assert owner.max_pending == 4
    assert 'service' not in cfg.protein_beam


def test_child_environment_queries_target_and_preserves_ld_priority(tmp_path):
    site = tmp_path / 'site-packages'
    nvrtc = site / 'nvidia/cuda_nvrtc/lib'
    cudnn = site / 'nvidia/cudnn/lib'
    empty = site / 'nvidia/empty/lib'
    for directory in (nvrtc, cudnn, empty):
        directory.mkdir(parents=True)
    (nvrtc / 'libnvrtc.so.12').touch()
    (cudnn / 'libcudnn.so.9').touch()
    target = tmp_path / 'target-python'
    target.write_text(
        "#!/bin/sh\nprintf '%s\\n' '"
        + json.dumps({'purelib': str(site), 'platlib': str(site)})
        + "'\n"
    )
    target.chmod(0o755)
    original = {
        'LD_LIBRARY_PATH': os.pathsep.join(('/driver', str(cudnn), '/driver')),
        'UNCHANGED': 'value',
    }

    runtime = lifecycle.af3_child_environment(target, original)

    assert runtime.library_directories == (
        str(nvrtc.resolve()),
        str(cudnn.resolve()),
    )
    assert runtime.environment['LD_LIBRARY_PATH'].split(os.pathsep) == [
        '/driver',
        str(cudnn),
        str(nvrtc.resolve()),
    ]
    assert runtime.environment['UNCHANGED'] == 'value'
    assert original['LD_LIBRARY_PATH'].split(os.pathsep) == [
        '/driver', str(cudnn), '/driver'
    ]


def test_owned_af3_constructs_final_child_env_before_popen(config,tmp_path,fake_command):
    priority = f'/owned-priority-{os.getpid()}'
    original = dict(os.environ, LD_LIBRARY_PATH=priority)
    owner = lifecycle.OwnedAF3(config, tmp_path, original)
    with owner as socket:
        assert call(socket, {'type': 'cpu_echo', 'value': 1})['value'] == 1

    state = json.loads(owner.state_path.read_text())
    mark = next(m for m in events(owner) if m['event'] == 'fake_model_load')
    assert Path(state['python_bin']).samefile(sys.executable)
    assert state['python_library_directories'] == list(owner.library_directories)
    assert mark['ld'] == owner.environment['LD_LIBRARY_PATH']
    assert mark['ld'].split(os.pathsep)[0] == priority
    assert mark['torch'] is False
    assert mark['jax_imported'] is False
    assert original['LD_LIBRARY_PATH'] == priority


def test_owner_separates_stdout_events_from_child_stderr(config,tmp_path,monkeypatch):
    script = FAKE.replace(
        "serve(lambda payload:payload,sys.argv[1])",
        """
sys.stderr.write('libprotobuf native diagnostic\\n')
sys.stderr.flush()
serve(lambda payload:payload,sys.argv[1])
""",
    )
    monkeypatch.setattr(
        lifecycle,
        '_owner_command',
        lambda config,socket:[sys.executable,'-u','-c',script,str(socket)],
    )
    owner = lifecycle.OwnedAF3(config, tmp_path, os.environ)
    with owner as socket:
        assert call(socket, {'type': 'cpu_echo', 'value': 2})['value'] == 2

    event_lines = owner.log_path.read_text().splitlines()
    assert event_lines
    assert all(line.startswith('{') for line in event_lines)
    assert all(isinstance(json.loads(line).get('event'), str) for line in event_lines)
    assert owner.diagnostics_path.read_text() == 'libprotobuf native diagnostic\n'
    state = json.loads(owner.state_path.read_text())
    assert state['log'] == str(owner.log_path)
    assert state['stdout_path'] == str(owner.log_path)
    assert state['stderr_path'] == str(owner.diagnostics_path)
    assert owner.log.closed and owner.stderr.closed


def test_startup_error_points_to_separate_diagnostics(config,tmp_path,monkeypatch):
    script = (
        "import sys;"
        "sys.stderr.write('fatal native diagnostic\\n');"
        "sys.stderr.flush();"
        "raise SystemExit(7)"
    )
    monkeypatch.setattr(
        lifecycle,
        '_owner_command',
        lambda config,socket:[sys.executable,'-u','-c',script],
    )
    owner = lifecycle.OwnedAF3(config, tmp_path, os.environ)
    with pytest.raises(RuntimeError) as raised:
        with owner:
            pytest.fail('Failed owner must not run the beam')
    assert str(owner.diagnostics_path) in str(raised.value)
    assert owner.diagnostics_path.read_text() == 'fatal native diagnostic\n'
    assert owner.log_path.read_text() == ''
    assert owner.log.closed and owner.stderr.closed


@pytest.mark.skipif(
    not Path('/alphafold3_venv/bin/python').is_file(),
    reason='licensed AF3 CPU image is not mounted',
)
def test_target_child_loads_nvrtc_and_runtime_compiled_cudnn_without_models():
    existing = os.pathsep.join((
        '/usr/local/nvidia/lib',
        '/usr/local/nvidia/lib64',
        '/.singularity.d/libs',
    ))
    runtime = lifecycle.af3_child_environment(
        '/alphafold3_venv/bin/python',
        {'LD_LIBRARY_PATH': existing},
    )
    assert runtime.environment['LD_LIBRARY_PATH'].startswith(existing + os.pathsep)
    assert any(path.endswith('/nvidia/cuda_nvrtc/lib') for path in runtime.library_directories)
    assert any(path.endswith('/nvidia/cudnn/lib') for path in runtime.library_directories)
    script = '''
import ctypes,json,sys
names=[
    "libnvrtc.so.12",
    "libnvrtc-builtins.so.12.6",
    "libcudnn.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
]
for name in names:
    ctypes.CDLL(name)
print(json.dumps({
    "loaded": names,
    "torch_imported": "torch" in sys.modules,
    "jax_imported": "jax" in sys.modules,
}))
'''
    completed = subprocess.run(
        [runtime.python_bin, '-I', '-B', '-c', script],
        env=runtime.environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result['loaded'] == [
        'libnvrtc.so.12',
        'libnvrtc-builtins.so.12.6',
        'libcudnn.so.9',
        'libcudnn_engines_runtime_compiled.so.9',
    ]
    assert result['torch_imported'] is False
    assert result['jax_imported'] is False


def test_private_owner_real_rpc_drain_and_gpu_environment(config,tmp_path,fake_command):
    original=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-test-only',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
    owner=lifecycle.OwnedAF3(config,tmp_path,original)
    with owner as socket:
        assert call(socket,{'type':'cpu_echo','value':42})['value']==42
        assert owner.process.poll() is None
        assert Path(socket).parent.stat().st_mode & 0o777==0o700
    state=json.loads(owner.state_path.read_text())
    assert state['status']=='drained' and state['shutdown_acknowledged']
    assert state['exit_code']==0 and not Path(socket).exists()
    marks=events(owner)
    assert len([m for m in marks if m['event']=='fake_model_load'])==1
    assert next(m for m in marks if m['event']=='fake_model_load')['cuda']=='GPU-test-only'
    assert next(m for m in marks if m['event']=='fake_model_load')['jax']=='cuda'
    assert any(m['event']=='shutdown_drained' for m in marks)
    assert original['JAX_PLATFORMS']=='cpu'


def test_owner_still_drains_when_cpu_work_raises(config,tmp_path,fake_command):
    owner=lifecycle.OwnedAF3(config,tmp_path,os.environ)
    with pytest.raises(RuntimeError,match='proposal failed'):
        with owner:
            raise RuntimeError('proposal failed')
    assert owner.process.poll()==0
    assert json.loads(owner.state_path.read_text())['status']=='drained'


@pytest.mark.parametrize('script,message',[("raise SystemExit(7)",'exited during startup'),('import time;time.sleep(600)','did not become ready')])
def test_startup_failure_retains_log_and_reaps_owned_process(config,tmp_path,monkeypatch,script,message):
    config.protein_beam.service.startup_timeout_s=.3
    monkeypatch.setattr(lifecycle,'_owner_command',lambda *args:[sys.executable,'-u','-c',script])
    owner=lifecycle.OwnedAF3(config,tmp_path,os.environ)
    with pytest.raises((RuntimeError,TimeoutError),match=message):
        with owner:pytest.fail('Failed owner must not run the beam')
    assert owner.process.poll() is not None
    state=json.loads(owner.state_path.read_text())
    assert state['status']=='startup_failed' and owner.log_path.exists()
    assert not Path(owner.socket_directory).exists()


def test_unresponsive_shutdown_is_bounded_and_fails_without_silent_success(config,tmp_path,monkeypatch):
    script=FAKE.replace("serve(lambda payload:payload,sys.argv[1])", """
import socket,time
sock=socket.socket(socket.AF_UNIX);sock.bind(sys.argv[1]);sock.listen();os.chmod(sys.argv[1],0o600)
mark('service_ready',socket=sys.argv[1]);time.sleep(600)
""")
    monkeypatch.setattr(lifecycle,'_owner_command',lambda config,socket:[sys.executable,'-u','-c',script,str(socket)])
    config.protein_beam.service.shutdown_timeout_s=.2
    owner=lifecycle.OwnedAF3(config,tmp_path,os.environ)
    with pytest.raises(RuntimeError,match='cleanup failed'):
        with owner:pass
    assert owner.process.poll() is not None
    assert json.loads(owner.state_path.read_text())['status']=='shutdown_failed'


def test_batch_auto_starts_one_model_and_completed_resume_starts_none(tmp_path,monkeypatch,fake_command):
    from astrolabe.interface import run_interface
    import astrolabe.protein_beam as beam
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-test-only')
    observed=[]
    def fake(cfg,destination,socket,**kwargs):
        assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['JAX_PLATFORMS']=='cpu'
        checkpoint=Path(destination)/'beam_state.json'
        if checkpoint.exists():return json.loads(checkpoint.read_text())
        assert call(socket,{'type':'cpu_echo','case':cfg.task_name})['case']==cfg.task_name
        observed.append(socket)
        checkpoint.parent.mkdir(parents=True)
        value=dict(status='complete',records=[{'id':0}],budget_complete=False)
        checkpoint.write_text(json.dumps(value));return value
    monkeypatch.setattr(beam,'run_beam',fake)
    args=dict(input_path=INPUT,target_homology=INPUT/'gd-ctd_data.json',output_root=tmp_path/'batch')
    assert run_interface(**args)==0
    assert len(observed)==3 and len(set(observed))==1
    owners=list((tmp_path/'batch/services').glob('*/lifecycle.json'))
    assert len(owners)==1 and json.loads(owners[0].read_text())['status']=='drained'
    monkeypatch.setattr(lifecycle,'OwnedAF3',lambda *a:pytest.fail('Completed recovery must not load a model'))
    assert run_interface(**args,resume=True)==0
    assert len(observed)==3


def test_explicit_service_is_reused_and_never_stopped(config,tmp_path,monkeypatch,fake_command):
    from astrolabe.interface import run_interface
    import astrolabe.protein_beam as beam
    def fake(cfg,destination,socket,**kwargs):
        assert call(socket,{'type':'cpu_echo','value':5})['value']==5
        return dict(status='complete',records=[],budget_complete=False)
    monkeypatch.setattr(beam,'run_beam',fake)
    owner=lifecycle.OwnedAF3(config,tmp_path/'external',os.environ)
    with owner as socket:
        run_interface(input_path=INPUT/'D5056_D15.pdb',target_homology=INPUT/'gd-ctd_data.json',output_root=tmp_path/'batch',socket_path=socket)
        assert owner.process.poll() is None
        assert not (tmp_path/'batch/services').exists()
        assert not any(m['event']=='shutdown_received' for m in events(owner))


def test_prepare_only_does_not_start_any_service(tmp_path,monkeypatch):
    from astrolabe.interface import run_interface
    monkeypatch.setattr(lifecycle,'OwnedAF3',lambda *a:pytest.fail('Preparation must not load any model'))
    assert run_interface(input_path=INPUT/'D5056_D15.pdb',target_homology=INPUT/'gd-ctd_data.json',output_root=tmp_path/'out',prepare_only=True)==0
    assert not (tmp_path/'out/services').exists()


def test_owned_parent_guard_rejects_wrong_parent(monkeypatch):
    monkeypatch.setenv('ASTROLABE_AF3_PARENT_PID',str(os.getppid()+1000000))
    with pytest.raises(RuntimeError,match='coordinator has already ended'):lifecycle.guard_owner_parent()


def test_cleanup_interrupt_reaps_before_recording_outcome(config,tmp_path,fake_command,monkeypatch):
    owner=lifecycle.OwnedAF3(config,tmp_path,os.environ)
    original=lifecycle.threading.Thread.join;called=[]
    def interrupt_once(thread,*args,**kwargs):
        if not called:
            called.append(True);raise KeyboardInterrupt('injected cleanup interrupt')
        return original(thread,*args,**kwargs)
    with pytest.raises(KeyboardInterrupt,match='injected'):
        with owner:
            monkeypatch.setattr(lifecycle.threading.Thread,'join',interrupt_once)
    assert owner.process.poll() is not None
    state=json.loads(owner.state_path.read_text())
    assert state['status']=='shutdown_failed' and state['cleanup_errors']
    assert not owner.socket_path.exists()


def test_pool_abort_callback_precedes_waiting_for_workers(monkeypatch):
    import astrolabe.protein_beam as beam
    order=[]
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):
            assert order==['owner_stopped'];order.append('pool_exited')
    monkeypatch.setattr(beam,'ProcessPoolExecutor',Pool)
    with pytest.raises(RuntimeError,match='injected worker error'):
        with beam.controlled_pool(lambda:order.append('owner_stopped')):
            raise RuntimeError('injected worker error')
    assert order==['owner_stopped','pool_exited']


def _stuck_worker():
    import time
    time.sleep(600)


def test_real_stuck_cpu_worker_is_reaped_on_error():
    import multiprocessing as mp
    import time
    from astrolabe.protein_beam import controlled_pool
    order=[];started=time.monotonic()
    with pytest.raises(RuntimeError,match='abort real pool'):
        with controlled_pool(lambda:order.append('owner_stopped'),max_workers=1,mp_context=mp.get_context('spawn')) as pool:
            future=pool.submit(_stuck_worker)
            time.sleep(.2)
            raise RuntimeError('abort real pool')
    assert order==['owner_stopped'] and time.monotonic()-started<15
    assert future.done()


def test_environment_restored_for_repeated_api_calls_and_errors(tmp_path,monkeypatch):
    from astrolabe.interface import run_interface,cpu_coordinator_environment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-original')
    monkeypatch.setenv('JAX_PLATFORMS','cuda')
    monkeypatch.setenv('JAX_PLATFORM_NAME','gpu')
    keys=['CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','JAX_PLATFORM_NAME','GH_CUDA_PREINITIALIZE']
    before={k:os.environ.get(k) for k in keys}
    with cpu_coordinator_environment() as captured:
        assert captured['CUDA_VISIBLE_DEVICES']=='GPU-original'
        assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['JAX_PLATFORMS']=='cpu'
        assert 'JAX_PLATFORM_NAME' not in os.environ
    args=dict(input_path=INPUT/'D5056_D15.pdb',target_homology=INPUT/'gd-ctd_data.json',prepare_only=True)
    for name in ['first','second']:
        run_interface(**args,output_root=tmp_path/name)
        assert {k:os.environ.get(k) for k in keys}==before
    with pytest.raises(FileNotFoundError):
        run_interface(input_path=tmp_path/'absent.pdb',target_homology=INPUT/'gd-ctd_data.json',output_root=tmp_path/'bad')
    assert {k:os.environ.get(k) for k in keys}==before


def test_recovery_rejects_live_owner_without_touching_it(config,tmp_path,fake_command):
    owner=lifecycle.OwnedAF3(config,tmp_path,os.environ)
    with owner:
        with pytest.raises(RuntimeError,match='still active'):lifecycle.recover_owned_services(tmp_path)
        assert owner.process.poll() is None and owner.socket_path.exists()


def test_dead_owned_socket_recovery_preserves_original_record(tmp_path):
    import socket,tempfile
    directory=Path(tempfile.mkdtemp(prefix='astrolabe-af3-'))
    path=directory/'owner.sock';sock=socket.socket(socket.AF_UNIX);sock.bind(str(path));sock.close()
    session=tmp_path/'services/previous';session.mkdir(parents=True)
    state=dict(pid=os.getpid(),pid_start_ticks='different-old-process',boot_id=lifecycle._boot_id(),
               directory_inode=directory.stat().st_ino,socket_inode=path.stat().st_ino,
               pid_namespace=Path('/proc/self/ns/pid').stat().st_ino,uid=os.getuid(),socket=str(path),status='ready')
    record=session/'lifecycle.json';record.write_text(json.dumps(state));original=record.read_bytes()
    lifecycle.recover_owned_services(tmp_path)
    assert record.read_bytes()==original and not directory.exists()
    assert json.loads((session/'recovery.json').read_text())['predictions_replayed']==0
    assert os.getpid()==state['pid']  # PID reuse never authorizes killing the current process.


def test_killed_coordinator_owner_exits_and_stale_socket_is_recoverable(tmp_path):
    import subprocess,time
    output=tmp_path/'job';ready=tmp_path/'ready.json'
    script=f'''
import json,os,time,sys
from pathlib import Path
from astrolabe.protein_beam import load_config
from astrolabe.evaluators import af3_lifecycle as lifecycle
lifecycle._owner_command=lambda config,socket:[sys.executable,'-u','-c',{FAKE!r},str(socket)]
owner=lifecycle.OwnedAF3(load_config({str(TASK)!r}),{str(output)!r},os.environ)
with owner as socket:
 Path({str(ready)!r}).write_text(json.dumps(dict(pid=owner.process.pid,socket=socket)))
 time.sleep(600)
'''
    coordinator=subprocess.Popen([sys.executable,'-c',script])
    try:
        deadline=time.monotonic()+15
        while not ready.exists() and time.monotonic()<deadline:
            assert coordinator.poll() is None
            time.sleep(.05)
        assert ready.exists()
        owner=json.loads(ready.read_text());coordinator.kill();coordinator.wait(timeout=3)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            identity=lifecycle._process_identity(owner['pid'])
            if identity is None or identity['state']=='Z':break
            time.sleep(.05)
        else:pytest.fail('Owned model outlived the killed coordinator')
        lifecycle.recover_owned_services(output)
        assert not Path(owner['socket']).exists()
    finally:
        if coordinator.poll() is None:
            coordinator.kill();coordinator.wait()


def test_sigint_delivered_to_background_thread_is_deferred_through_reap(config,tmp_path,monkeypatch):
    import signal,threading,time
    script=FAKE.replace("serve(lambda payload:payload,sys.argv[1])", """
import socket,time
sock=socket.socket(socket.AF_UNIX);sock.bind(sys.argv[1]);sock.listen();os.chmod(sys.argv[1],0o600)
mark('service_ready',socket=sys.argv[1]);time.sleep(600)
""")
    monkeypatch.setattr(lifecycle,'_owner_command',lambda config,socket:[sys.executable,'-u','-c',script,str(socket)])
    config.protein_beam.service.shutdown_timeout_s=.2
    owner=lifecycle.OwnedAF3(config,tmp_path,os.environ)
    done=threading.Event();background=threading.Thread(target=done.wait);background.start()
    original=owner._force_reap
    def reap_with_signal():
        signal.pthread_kill(background.ident,signal.SIGINT)
        time.sleep(.05)  # Signal handler runs while the owner is still alive.
        original()
    monkeypatch.setattr(owner,'_force_reap',reap_with_signal)
    # This test requires an interrupting caller. Other application tests install
    # the CLI's graceful-stop handler, which has different deliberate semantics.
    previous_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            with owner:pass
        assert owner.process.poll() is not None
        assert json.loads(owner.state_path.read_text())['status']=='shutdown_failed'
        assert not owner.socket_path.exists()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        done.set();background.join(timeout=2)


def test_recovery_checks_live_starting_owner_even_without_socket(tmp_path):
    session=tmp_path/'services/starting';session.mkdir(parents=True)
    state=dict(pid=os.getpid(),pid_start_ticks=lifecycle._process_identity(os.getpid())['start_ticks'],
               boot_id=lifecycle._boot_id(),pid_namespace=Path('/proc/self/ns/pid').stat().st_ino,
               directory_inode=0,uid=os.getuid(),socket=str(tmp_path/'not-bound-yet/owner.sock'),status='starting')
    (session/'lifecycle.json').write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match='still active'):lifecycle.recover_owned_services(tmp_path)
