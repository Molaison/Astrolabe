import math
import numpy as np
import pytest
from omegaconf import OmegaConf
from astrolabe.core import global_config
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.random_substitution import generate_aa_freqs
from test_mpnn_sync_policy import setup_case


def test_same_config_policy_is_consumed_by_mcmc_and_mpnn_builder_and_mutate(tmp_path, monkeypatch):
    config = {'allowed_residues':['C','D'], 'bias':{'C':math.log(2)}, 'temperature':2.0, 'max_mutations':1}
    protomers, scores, mpnn = setup_case(tmp_path, monkeypatch, 'probability_mean', config)
    mcmc_cfg = OmegaConf.create({'enabled_aliases':['random'],
        'mutator_terms':{'random':{'_type':'MCMC','proposal_policy':config}},
        'MCMC':{'mutation_method':'random','select_positions':'random'}})
    mcmc = build_mutators(mcmc_cfg, tmp_path)['random']
    mcmc.mutation_weight = 1
    global_config.set_variable('mutation_num', 2)
    runtime = {'oligo':['AB','AC'],'ignored_complexes':[],'negative_design_complexes':[]}
    segment = global_config.segment_manager.segments['A_1']
    fixed = global_config.segment_manager.segments['A_2']
    raw = generate_aa_freqs(3)[0, 0]
    c_idx, d_idx = list(segment.aa_order).index('C'), list(segment.aa_order).index('D')
    np.random.seed(17)
    mcmc.mutate(protomers, scores, dict(runtime))
    mcmc_probs = segment.probs[0][0]
    assert mcmc_probs[c_idx] / mcmc_probs[d_idx] == pytest.approx(math.sqrt(raw[c_idx] / raw[d_idx]) * math.sqrt(2))
    assert sum(len(seg.mutations or []) for seg in global_config.segment_manager.unique_segments.values()) == 1
    assert fixed.sequence == 'C' and not fixed.mutations
    global_config.segment_manager.clear_mutations()
    np.random.seed(17)
    mpnn.mutate(protomers, scores, dict(runtime))
    mpnn_probs = segment.probs[0][0]
    assert mpnn_probs[c_idx] == pytest.approx(.5) and mpnn_probs[d_idx] == pytest.approx(.5)
    assert sum(len(seg.mutations or []) for seg in global_config.segment_manager.unique_segments.values()) == 1
    assert fixed.sequence == 'C' and not fixed.mutations
    global_config.reset()


def test_native_joint_rejects_probability_transforms_but_accepts_budget(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='native joint decoding'):
        setup_case(tmp_path, monkeypatch, 'native_tie', {'temperature':.5})
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, 'native_tie', {'max_mutations':0})
    mutator.mutate(protomers, scores, {'oligo':['AB','AC'], 'ignored_complexes':[], 'negative_design_complexes':[]})
    assert all(not segment.mutations for segment in global_config.segment_manager.unique_segments.values())
    global_config.reset()


def test_configured_policy_with_real_cpu_mpnn_wrapper(tmp_path, monkeypatch):
    import torch
    from astrolabe.mutators.wrappers.ligandmpnn_utils import ProteinMPNN
    from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN
    from test_ligandmpnn_device_binding import _args
    torch.manual_seed(501)
    state = ProteinMPNN(k_neighbors=8, model_type='protein_mpnn').state_dict()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.torch.load',
                        lambda *_a, **_kw: {'num_edges':8,'model_state_dict':state})
    args = _args(tmp_path)
    args.model_type = 'protein_mpnn'
    args.temperature = 1.0
    runner = LigandMPNN(args)
    cfg = {'allowed_residues':['C','D'], 'bias':{'C':.5}, 'temperature':2.0, 'max_mutations':1}
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, 'probability_mean', cfg)
    mutator.runner = runner
    global_config.set_variable('mutation_num', 2)
    manager = global_config.segment_manager
    segment = manager.segments['A_1']
    np.random.seed(5)
    mutator.mutate(protomers, scores, {'oligo':['AB','AC'], 'ignored_complexes':[], 'negative_design_complexes':[]})
    probability = segment.probs[0][0]
    nonzero = {str(aa) for aa, prob in zip(segment.aa_order, probability) if prob > 0}
    assert nonzero <= {'C','D'} and nonzero
    assert sum(len(seg.mutations or []) for seg in manager.unique_segments.values()) <= 1
    assert not manager.segments['A_2'].mutations
    global_config.reset()
