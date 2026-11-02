import json

import numpy as np
import pytest

from astrolabe.evaluators.tree_codec import dump_tree, load_tree


def test_rejects_input_before_archive_write(tmp_path, monkeypatch):
    monkeypatch.setattr(np, 'savez', lambda *a, **kw: pytest.fail('oversize data reached archive writer'))
    with pytest.raises(ValueError, match='budget'):
        dump_tree(tmp_path, {'pair': np.zeros((16, 16), dtype=np.float32)}, max_bytes=128)
    assert not (tmp_path / 'arrays.npz').exists()


def test_archive_budget_is_checked_before_numpy_load(tmp_path, monkeypatch):
    dump_tree(tmp_path, {'pair': np.arange(256, dtype=np.float32)})
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('oversize data reached numpy loader'))
    with pytest.raises(ValueError, match='budget'):
        load_tree(tmp_path, max_bytes=128)


def test_real_archive_header_must_match_manifest_before_allocation(tmp_path, monkeypatch):
    dump_tree(tmp_path, np.ones((2, 2), dtype=np.float32))
    manifest = tmp_path / 'tree.json'
    value = json.loads(manifest.read_text())
    value['tree'].update(shape=[1], nbytes=4)
    manifest.write_text(json.dumps(value))
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('invalid header reached array loader'))
    with pytest.raises(ValueError, match='header'):
        load_tree(tmp_path)


def test_metadata_and_object_array_budgets(tmp_path):
    with pytest.raises(ValueError, match='budget'):
        dump_tree(tmp_path, np.array(['A' * 1024], dtype=object), max_metadata_bytes=256)
    dump_tree(tmp_path, {'strings': np.array(['A', 'B'], dtype=object), 'opaque': b'abcd'})
    result = load_tree(tmp_path)
    np.testing.assert_array_equal(result['strings'], ['A', 'B'])
    assert result['opaque'] == b'abcd'
    with pytest.raises(ValueError, match='metadata'):
        load_tree(tmp_path, max_metadata_bytes=16)


def test_cycles_and_duplicate_archive_members_fail_closed(tmp_path):
    recursive = []
    recursive.append(recursive)
    with pytest.raises(ValueError, match='cyclic'):
        dump_tree(tmp_path, recursive)
    import zipfile
    dump_tree(tmp_path, np.arange(3))
    with zipfile.ZipFile(tmp_path / 'arrays.npz', 'a') as archive:
        with pytest.warns(UserWarning, match='Duplicate'):
            archive.writestr('a0.npy', b'duplicate')
    with pytest.raises(ValueError, match='duplicate'):
        load_tree(tmp_path)


def test_oversized_json_is_rejected_before_parse(tmp_path, monkeypatch):
    (tmp_path / 'tree.json').write_bytes(b' ' * 257)
    monkeypatch.setattr(json, 'loads', lambda *a, **kw: pytest.fail('oversized metadata reached JSON parser'))
    with pytest.raises(ValueError, match='metadata'):
        load_tree(tmp_path, max_metadata_bytes=256)


def test_nonfinite_json_numbers_are_rejected(tmp_path):
    dump_tree(tmp_path, 0.0)
    for number in ('NaN', 'Infinity', '1e999'):
        (tmp_path / 'tree.json').write_text(
            '{"version":1,"tree":{"t":"scalar","v":' + number + '}}')
        with pytest.raises(ValueError, match='nonfinite'):
            load_tree(tmp_path)
