"""Budget wiring, native controller/quota bounds and immutable recovery; no models."""
import copy
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf, open_dict
import pytest
from branching_test_support import native_control_assets

from astrolabe import interface
from astrolabe.protein_beam import load_config

INPUT = Path('/case/inputs/001_binder_interface/01_astrolabe_beam/03_bind_target_homologies')


def prepare(output, budget=None, directory=False):
    return interface._prepare(INPUT if directory else INPUT/'D5056_D15.pdb',
                              INPUT/'gd-ctd_data.json', output, None, None, budget)


@pytest.mark.parametrize('budget', [1, 4, 13])
def test_budget_is_per_complex_and_preserves_scientific_settings(tmp_path, budget):
    baseline, defaults = prepare(tmp_path, directory=True)
    plan, artifacts = prepare(tmp_path, budget, directory=True)
    assert len(plan['cases']) == 3
    for case in plan['cases']:
        cfg = OmegaConf.create(artifacts[case['config']])
        original = OmegaConf.create(defaults[case['config']])
        assert cfg.protein_beam.max_evaluations == budget
        assert cfg.protein_beam.max_generations is None
        assert cfg.protein_beam.af3_call_budget.max_total == budget
        assert cfg.protein_beam.af3_call_budget.max_per_case == budget
        assert original.protein_beam.max_evaluations == 7
        assert original.protein_beam.max_generations == 1
        # The entire remainder must match, including geometry, MPNN, seed,
        # recycles, homology and the five losses, rather than selected examples.
        for key in ['max_evaluations', 'max_generations', 'af3_call_budget']:
            cfg.protein_beam[key] = copy.deepcopy(original.protein_beam[key])
        assert OmegaConf.to_container(cfg) == OmegaConf.to_container(original)


@pytest.mark.parametrize('budget', [0, -1, 1.5, True, '3'])
def test_invalid_budget_never_creates_output(tmp_path, budget):
    out = tmp_path/'out'
    with pytest.raises(ValueError, match='positive integer'):
        interface.run_interface(input_path=INPUT/'D5056_D15.pdb',
                                target_homology=INPUT/'gd-ctd_data.json',
                                output_root=out, prepare_only=True, budget=budget)
    assert not out.exists()


def test_same_budget_resume_preserves_artifacts_changed_or_omitted_budget_is_rejected(tmp_path, monkeypatch):
    from astrolabe.evaluators import af3_lifecycle
    monkeypatch.setattr(af3_lifecycle, 'OwnedAF3', lambda *a, **k: pytest.fail('No AF3 owner'))
    args = dict(input_path=INPUT/'D5056_D15.pdb', target_homology=INPUT/'gd-ctd_data.json',
                output_root=tmp_path/'out', prepare_only=True)
    assert interface.run_interface(**args, budget=13) == 0
    def digests():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (tmp_path/'out').rglob('*') if p.is_file()}
    before = digests()
    assert interface.run_interface(**args, budget=13, resume=True) == 0
    for changed in [None, 7, 14]:
        with pytest.raises(ValueError, match='configuration changed'):
            interface.run_interface(**args, budget=changed, resume=True)
        assert digests() == before


def test_native_cli_passes_budget_to_interface(monkeypatch, tmp_path):
    from astrolabe import cli
    received = []
    monkeypatch.setattr(interface, 'run_interface', lambda **k: received.append(k) or 0)
    with pytest.raises(SystemExit) as error:
        cli.main(['refine', 'interface', '--input', 'complex.pdb',
                  '--target-homology', 'target.json', '--out-dir', str(tmp_path),
                  '--budget', '13', '--prepare-only'])
    assert error.value.code == 0
    assert received[0]['budget'] == 13
    assert received[0]['prepare_only']


@pytest.mark.parametrize('budget', ['0', '-1', '1.5', 'word'])
def test_native_cli_rejects_invalid_budget_before_handler(monkeypatch, budget):
    from astrolabe import cli
    monkeypatch.setattr(interface, 'run_interface', lambda **k: pytest.fail('Invalid budget entered handler'))
    with pytest.raises(SystemExit) as error:
        cli.main(['refine', 'interface', '--input', 'complex.pdb',
                  '--target-homology', 'target.json', '--out-dir', 'out',
                  '--budget', budget, '--prepare-only'])
    assert error.value.code == 2


def simulated_beam(tmp_path, monkeypatch, budget, duplicates=False):
    """Real coordinator/quota with synthetic CPU worker results, never AF3."""
    import astrolabe.protein_beam as beam
    from astrolabe.protein_beam_budget import claim_call
    output = tmp_path/'out'
    plan, artifacts = prepare(output, budget)
    for relative, text in artifacts.items():
        p = output/relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    cfg = load_config(output/plan['cases'][0]['config'])
    with open_dict(cfg):
        # Geometry itself is separately verified by the real-structure suite.
        # Synthetic worker records do not pretend to be physical predictions.
        cfg.protein_beam.root_calibration.enabled = False
        cfg.protein_beam.gates = {}
        cfg.protein_beam.selection = 'weighted'
        cfg.protein_beam.holes_ranking = 'raw_sum'
    root = str(cfg.seqs.A)
    evaluations = []
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, task):
            if task['kind'] == 'proposal':
                seq = list(root)
                if not duplicates:
                    i = task['id'] % len(seq)
                    seq[i] = 'Y' if seq[i] != 'Y' else 'A'
                result = dict(task, sequence=''.join(seq))
            else:
                claim_call(cfg.protein_beam.af3_call_budget, task)
                evaluations.append(task['id'])
                beam_output = Path(cfg.out_dir)
                candidate = beam_output/'candidates'/f"{task['id']:06d}"
                candidate.mkdir(parents=True, exist_ok=True)
                snapshot = candidate/'scores.pkl'
                snapshot.write_bytes(f"synthetic-scores-{task['id']}".encode())
                structure = beam_output/'workers'/f"fixture-{task['id']}"/'structures'/f"step_{task['id']:03d}_AB_model.cif"
                structure.parent.mkdir(parents=True, exist_ok=True)
                structure.write_text(f"data_fixture_{task['id']}\n")
                result = dict(task, status='done', score=float(task['id']),
                              metrics={m: 0.2 for m in beam.METRICS},
                              holes={}, geometry={},
                              snapshot=str(snapshot), structure_path=str(structure))
                from astrolabe.protein_beam_contract import native_artifact_manifest
                result['native_artifacts'] = native_artifact_manifest(result)
                (candidate/'result.json').write_text(json.dumps(result))
            future = Future()
            future.set_result(result)
            return future
    monkeypatch.setattr(beam, 'ProcessPoolExecutor', Pool)
    state = beam.run_beam(cfg, cfg.out_dir, 'synthetic', workers=2, in_flight=2)
    return beam, cfg, state, evaluations


@pytest.mark.parametrize('budget', [1, 4, 13])
@pytest.mark.usefixtures("native_control_assets")
def test_native_coordinator_and_issued_quota_honor_budget_across_generations(tmp_path, monkeypatch, budget):
    from astrolabe.protein_beam_budget import reserve_calls
    beam, cfg, state, evaluations = simulated_beam(tmp_path, monkeypatch, budget)
    assert evaluations == list(range(budget))
    assert state['budget_complete']
    assert len(state['records']) == budget
    assert state['generation'] == (0 if budget == 1 else 1 if budget <= 7 else 2)
    ledger = json.loads(Path(cfg.protein_beam.af3_call_budget.ledger).read_text())
    assert len(ledger['entries']) == budget
    assert all(e['status'] == 'issued' for e in ledger['entries'].values())
    extra = dict(id=budget, kind='evaluate', seed=cfg.seed)
    assert reserve_calls(cfg.protein_beam.af3_call_budget, [extra]) == ([], [extra])
    beam.run_beam(cfg, cfg.out_dir, 'synthetic', workers=2, in_flight=2, resume=True)
    assert len(evaluations) == budget


@pytest.mark.usefixtures("native_control_assets")
def test_duplicate_exhaustion_stops_below_budget_without_spending_remaining_quota(tmp_path, monkeypatch):
    _, cfg, state, evaluations = simulated_beam(tmp_path, monkeypatch, 13, duplicates=True)
    assert evaluations == [0]
    assert not state['budget_complete']
    assert state['stop_reason'] == 'distinct_sequence_proposal_cap'
    ledger = json.loads(Path(cfg.protein_beam.af3_call_budget.ledger).read_text())
    assert len(ledger['entries']) == 1
