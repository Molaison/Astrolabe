from pathlib import Path
import pytest
from branching_test_support import native_control_assets
from astrolabe.protein_beam import load_config
from astrolabe.protein_beam_contract import scientific_contract

pytestmark = pytest.mark.usefixtures("native_control_assets")


def test_same_path_homology_content_change_invalidates_contract(tmp_path):
    cfg=load_config(str(Path(__file__).parent / 'data/protein_beam/02_fixture.yaml'))
    p=tmp_path/'homology.json';p.write_text('{"value":1}')
    cfg.homologies={'B':str(p)+':B'}
    first,hash1=scientific_contract(cfg)
    p.write_text('{"value":2}')
    second,hash2=scientific_contract(cfg)
    assert first['data_sha256']!=second['data_sha256'] and hash1!=hash2
    assert first['source_sha256']==second['source_sha256']
    assert first['readonly_weight_asset_stat']


def test_completed_native_artifacts_are_bound_to_content(tmp_path):
    from astrolabe.protein_beam_contract import native_artifact_manifest, verify_native_artifacts
    snapshot = tmp_path / "scores.pkl"
    structure = tmp_path / "model.cif"
    snapshot.write_bytes(b"owned native snapshot")
    structure.write_text("data_native\n")
    manifest = native_artifact_manifest({"snapshot": str(snapshot), "structure_path": str(structure)})
    verify_native_artifacts(manifest)
    snapshot.write_bytes(b"changed native snapshot")
    with pytest.raises(ValueError, match="content changed"):
        verify_native_artifacts(manifest)
