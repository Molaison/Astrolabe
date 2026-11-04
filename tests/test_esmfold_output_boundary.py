"""Real torch CPU/native-output parity; no weights, CUDA, or broker certification."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
import weakref

import numpy as np
import pytest
import torch

from astrolabe.evaluators.esmfold_output import SCORE_FIELDS, resolve_pdb_boundary
from astrolabe.evaluators.lifecycle import cpu_tree
from astrolabe.evaluators.torch_rng import capture, restore
from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.wrappers import esmfold_daemon


@pytest.fixture
def native():
    path = Path(__file__).parent / 'fixtures/esmfold_output_native.py'
    spec = importlib.util.spec_from_file_location('_native_esmfold_output_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_output(native, batch=2, length=24, dtype=torch.float32, padded=False):
    generator = torch.Generator().manual_seed(1009)
    output = {'aatype': torch.arange(length).remainder(21).expand(batch, length).clone()}
    native.make_atom14_masks(output)
    chain = torch.div(torch.arange(length), length//2, rounding_mode='floor').clamp(max=1)
    output.update({
        'positions': torch.rand((2,batch,length,14,6), generator=generator, dtype=dtype)[..., ::2],
        'residue_index': (torch.arange(length)+512*chain).expand(batch,length).clone(),
        'chain_index': chain.expand(batch,length).clone(),
        'plddt': torch.rand((batch,length,74), generator=generator, dtype=dtype)[..., ::2]*100,
        'mean_plddt': torch.rand((batch,), generator=generator, dtype=dtype)*100,
        'ptm': torch.rand((batch,), generator=generator, dtype=dtype),
        'predicted_aligned_error': torch.rand((batch,length,length), generator=generator, dtype=dtype),
        's_z': torch.ones((batch,length,length,128), dtype=dtype),
    })
    if padded:
        output['atom37_atom_exists'][:, -2:] = 0
        output['chain_index'][:, -2:] = -1
    return output


def assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_fixture_integrity_and_actual_source_gate(native):
    path = Path(native.__file__)
    manifest = json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['fixture_sha256']
    assert manifest['esm_ref'] == '43c67d952da30d2471f609848931d20ec8c98532'
    assert manifest['openfold_ref'] == '8396098268a5675b0ff776ec274f15d380a4ccf5'
    assert resolve_pdb_boundary(native.NativeModel()) is not None


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('padded', [False, True])
def test_native_pdb_score_dtype_atom_order_and_rng_exact(native, batch, dtype, padded):
    output = make_output(native, batch=batch, dtype=dtype, padded=padded)
    original = {key: value.clone() for key, value in output.items()}
    state = capture('cpu')
    expected_pdbs = native.output_to_pdb(output)
    expected_scores = cpu_tree({key: output[key] for key in SCORE_FIELDS})
    expected_state = capture('cpu')
    boundary = resolve_pdb_boundary(native.NativeModel())
    host = boundary.prepare_output(output)
    assert boundary.finalize_pdb(host) == expected_pdbs
    assert_tree_equal({key: host[key] for key in SCORE_FIELDS}, expected_scores)
    assert_tree_equal(capture('cpu'), expected_state)
    assert_tree_equal(expected_state, state)
    assert_tree_equal(output, original)
    assert all(value.device.type == 'cpu' for value in host.values())
    assert 's_z' not in host and 'positions' not in host


def test_missing_chain_preserves_native_error(native):
    output = make_output(native)
    del output['chain_index']
    boundary = resolve_pdb_boundary(native.NativeModel())
    with pytest.raises(AttributeError) as expected:
        native.output_to_pdb(output)
    with pytest.raises(type(expected.value), match='astype') as actual:
        boundary.finalize_pdb(boundary.prepare_output(output))
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize('symbol', ['output_to_pdb', 'atom14_to_atom37', 'batched_gather',
                                    'to_pdb', 'OFProtein', 'get_pdb_headers', '_chain_end'])
def test_modified_native_dependency_is_not_reused(native, monkeypatch, symbol):
    def unknown(*args, **kwargs):
        raise AssertionError('unknown function must not be called by the audit')
    monkeypatch.setattr(native, symbol, unknown)
    assert resolve_pdb_boundary(native.NativeModel()) is None


def test_unknown_model_wrapper_and_wrapped_native_are_not_reused(native, monkeypatch):
    class Custom(native.NativeModel):
        def output_to_pdb(self, output):
            return ['custom output']
    assert resolve_pdb_boundary(Custom()) is None
    monkeypatch.setattr(native.output_to_pdb, '__wrapped__', native.output_to_pdb, raising=False)
    assert resolve_pdb_boundary(native.NativeModel()) is None


def test_changed_atom_names_are_not_reused(native, monkeypatch):
    monkeypatch.setattr(native.residue_constants, 'atom_types', list(reversed(native.atom_types)))
    assert resolve_pdb_boundary(native.NativeModel()) is None


def test_host_finalizer_never_copies_or_accepts_device_tensors(native, monkeypatch):
    boundary = resolve_pdb_boundary(native.NativeModel())
    host = boundary.prepare_output(make_output(native))
    def forbidden(*args, **kwargs):
        raise AssertionError('host finalizer attempted a tensor transfer')
    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden)
    monkeypatch.setattr(torch.Tensor, 'to', forbidden)
    assert len(boundary.finalize_pdb(host)) == 2
    host['plddt'] = torch.empty(host['plddt'].shape, device='meta')
    with pytest.raises(ValueError, match='requires CPU tensors'):
        boundary.finalize_pdb(host)


def test_materialization_visits_only_required_fields_once(native, monkeypatch):
    boundary = resolve_pdb_boundary(native.NativeModel())
    output = make_output(native)
    original_cpu = torch.Tensor.cpu
    copies = []
    def observed_cpu(tensor, *args, **kwargs):
        copies.append((tensor.data_ptr(), tensor.numel()*tensor.element_size()))
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', observed_cpu)
    host = boundary.prepare_output(output)
    assert len(copies) == len(host) == 9
    assert len({pointer for pointer, _ in copies}) == len(copies)
    assert output['s_z'].data_ptr() not in {pointer for pointer, _ in copies}
    expected = sum(output[key].numel()*output[key].element_size() for key in (
        'aatype', 'atom37_atom_exists', 'residue_index', *SCORE_FIELDS))
    expected += host['final_atom_positions'].numel()*host['final_atom_positions'].element_size()
    assert sum(size for _, size in copies) == expected


def _worker_handle(scope):
    # Execute the actual production handler, replacing only infrastructure
    # dependencies. This is a contract check, not a running daemon/broker test.
    tree = ast.parse(inspect.getsource(esmfold_daemon))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    handle = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'handle')
    exec(compile(ast.Module(body=[handle], type_ignores=[]), esmfold_daemon.__file__, 'exec'), scope)
    return scope['handle']


@pytest.mark.parametrize('custom', [False, True])
def test_actual_worker_boundary_output_rng_and_unused_lifetime(native, tmp_path, custom):
    state = {'lease': False}
    unused = []
    class Model(native.NativeModel):
        def infer(self, sequences):
            random.random(); np.random.random(); torch.rand(3)
            output = make_output(native)
            unused.append(weakref.ref(output['s_z']))
            return output
    if custom:
        def custom_pdb(self, output):
            # Unknown vendor code may consume RNG; preserve it inside the lease
            # before RNG capture and return its exact custom result.
            return [str(random.random()), str(torch.rand(1).item())]
        Model.output_to_pdb = custom_pdb
    model = Model()
    boundary = resolve_pdb_boundary(model)
    assert (boundary is None) is custom
    initial = capture('cpu')
    reference_output = model.infer(['AA', 'CC'])
    expected = {'pdbs': model.output_to_pdb(reference_output),
                'output': cpu_tree({key: reference_output[key] for key in SCORE_FIELDS}),
                'rng': capture('cpu')}
    del reference_output
    request = tmp_path / 'request-output'
    request.mkdir(mode=0o700)
    dump_tree(request / 'input', {'sequences': ['AA','CC'], 'rng': initial})
    events = []
    @contextlib.contextmanager
    def operation(*args, **kwargs):
        assert kwargs['phase'] == 'device'
        state['lease'] = True
        class Lease:
            def update_lifecycle(self, **kwargs):
                assert state['lease']
        try:
            yield Lease()
        finally:
            assert unused[-1]() is None, 'unused device output retained past lease'
            state['lease'] = False
    watched = {native.atom14_to_atom37.__code__: 'map_atoms', native.to_pdb.__code__: 'format_pdb',
               capture.__code__: 'capture_rng', restore.__code__: 'restore_rng'}
    if custom:
        watched[Model.output_to_pdb.__code__] = 'custom_pdb'
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in watched:
            events.append((watched[frame.f_code], state['lease']))
    handle = _worker_handle(dict(
        request_directory=lambda value: Path(value), load_tree=load_tree, dump_tree=dump_tree,
        contextlib=contextlib, sys=sys, torch=torch, operation=operation, actual={},
        restore=restore, capture=capture, spec={'device':{'name':'cpu'}}, model=model,
        pdb_boundary=boundary, SCORE_FIELDS=SCORE_FIELDS, cpu_tree=cpu_tree, lifecycle=lambda: {},
    ))
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        assert handle({'request_dir': str(request)}) == {'request': request.name}
    finally:
        sys.setprofile(previous)
        restore(initial, 'cpu')
    assert_tree_equal(load_tree(request / 'output'), expected)
    assert ('capture_rng', True) in events and ('restore_rng', True) in events
    if custom:
        assert ('custom_pdb', True) in events
    else:
        assert ('map_atoms', True) in events and ('map_atoms', False) not in events
        assert ('format_pdb', False) in events and ('format_pdb', True) not in events
