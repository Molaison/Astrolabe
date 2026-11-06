"""Failure injection: successful zero-valued metrics differ from failed scoring."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from astrolabe.losses import interaction


def test_strict_shape_does_not_convert_runtime_failure_to_zero(monkeypatch):
    monkeypatch.setattr(interaction,'_require_pyrosetta',lambda:None)
    monkeypatch.setattr(interaction,'_isolate_chains',lambda p,c:NS(total_residue=lambda:2,num_chains=lambda:2))
    class Filter:
        def jump_id(self,n):pass
        def use_rosetta_radii(self,x):pass
        def report_sm(self,p):raise RuntimeError('injected surface failure')
    monkeypatch.setattr(interaction,'ShapeComplementarityFilter',Filter)
    with pytest.raises(RuntimeError,match='injected'):interaction._compute_shape_complementarity_between_chains(object(),'A','B',strict=True)
    assert interaction._compute_shape_complementarity_between_chains(object(),'A','B') is None


@pytest.mark.parametrize('value',[0.,float('nan'),-1.])
def test_strict_shape_allows_real_zero_but_rejects_invalid_values(monkeypatch,value):
    monkeypatch.setattr(interaction,'_require_pyrosetta',lambda:None)
    monkeypatch.setattr(interaction,'_isolate_chains',lambda p,c:NS(total_residue=lambda:2,num_chains=lambda:2))
    class Filter:
        def jump_id(self,n):pass
        def use_rosetta_radii(self,x):pass
        def report_sm(self,p):return value
    monkeypatch.setattr(interaction,'ShapeComplementarityFilter',Filter)
    if np.isfinite(value) and value>=0:
        assert interaction._compute_shape_complementarity_between_chains(object(),'A','B',strict=True)==0.
    else:
        with pytest.raises(ValueError):interaction._compute_shape_complementarity_between_chains(object(),'A','B',strict=True)


@pytest.mark.parametrize('values',[(float('nan'),0.),(0.,float('inf')),(float('inf'),float('inf')),(1e308,-1e308)])
def test_buns_nonfinite_counts_and_difference_cannot_clip_to_zero(monkeypatch,values):
    readings=iter(values)
    filt=NS(report_sm=lambda pose:next(readings))
    fake=NS(rosetta=NS(protocols=NS(rosetta_scripts=NS(XmlObjects=NS(static_get_filter=lambda xml:filt))),
        core=NS(id=NS(AtomID=lambda a,r:(a,r))),numeric=NS(xyzVector_double_t=lambda *x:np.asarray(x))))
    monkeypatch.setattr(interaction,'pyrosetta',fake)
    class Pose:
        def total_residue(self):return 2
        def pdb_info(self):return NS(chain=lambda i:'AB'[i-1])
        def clone(self):return self
        def residue(self,i):return NS(natoms=lambda:1)
        def xyz(self,x):return np.zeros(3)
        def set_xyz(self,*args):pass
        def update_residue_neighbors(self):pass
    with pytest.raises(ValueError,match='finite'):interaction._compute_interface_buried_unsatisfied_hbonds(Pose(),['A'],['B'])


def test_pae_requires_complete_square_residue_coverage():
    from astrolabe.protein_beam import validate_pae_matrices
    for value in [np.zeros((270,270)),np.zeros((271,270)),np.zeros(271),np.full((271,271),np.nan)]:
        with pytest.raises(ValueError):validate_pae_matrices(value,271)
    validate_pae_matrices(np.zeros((1,271,271)),271)
    validate_pae_matrices(np.zeros((2,384,384)),271)


def test_native_chain_summary_resolves_to_complete_residue_confidence(tmp_path,monkeypatch):
    import json
    from astrolabe.losses import confidence
    from astrolabe.protein_beam import validate_pae_matrices
    folder=tmp_path/'alphafold3/step_000_AB';folder.mkdir(parents=True)
    raw=np.asarray([[0.,1.,2.],[3.,0.,4.],[5.,6.,0.]])
    (folder/'step_000_AB_confidences.json').write_text(json.dumps(dict(
        token_res_ids=[1,2,1],token_chain_ids=['A','A','B'],pae=raw.tolist())))
    monkeypatch.setattr(confidence,'_maybe_get_global_config_value',lambda key:tmp_path if key=='raw_model_results_dir' else 0)
    scores=NS(pae={'AB':[np.asarray([2.,3.])]})
    full=confidence._resolve_residue_level_pae_matrices(scores,oligo_name='AB')
    validate_pae_matrices(full,3)
    assert np.asarray(full).shape==(1,3,3) and np.allclose(full[0],raw)
