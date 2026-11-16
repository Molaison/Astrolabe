from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.evaluators.preparation_cache import (
    AF3CcdCache, ContentCache, CopyOnAccessMapping, FileObjectCache, MsaRowCache,
    install_msa_row_dispatch, msa_row_cache_scope,
)


def encoder(rows, polymer):
    """Small observable pure encoder; native parity is qualified separately."""
    offset = {"protein": 0, "rna": 100}[polymer]
    size = len("".join(c for c in rows[0] if not c.islower())) if rows else 0
    encoded, deletions = [], []
    for row in rows:
        values, gaps, gap = [], [], 0
        for char in row:
            if char.islower():
                gap += 1
            else:
                if char == "?":
                    raise ValueError("invalid residue")
                values.append(ord(char) + offset)
                gaps.append(gap)
                gap = 0
        if len(values) != size:
            raise ValueError("different row lengths")
        encoded.append(values)
        deletions.append(gaps)
    return (np.array(encoded, dtype=np.int32).reshape(len(rows), size),
            np.array(deletions, dtype=np.int32).reshape(len(rows), size))


def test_msa_delta_encodes_only_changed_rows_and_does_not_share_storage():
    cache = MsaRowCache()
    base = ["ACDE", "AcCDE", "AC-E", "AAAA"]
    first = cache.extract(encoder, base, "protein")
    assert cache.encoded_rows == 4
    first[0][1, 0] = -100
    mutant = ["ACDF", *base[1:]]
    got = cache.extract(encoder, mutant, "protein")
    expected = encoder(mutant, "protein")
    for actual, native in zip(got, expected):
        np.testing.assert_array_equal(actual, native)
    assert cache.encoded_rows == 5
    assert cache.hits == 3
    assert cache.extract(encoder, base, "protein")[0][1, 0] == ord("A")


@pytest.mark.parametrize("mutant,polymer", [
    (["ACDE", "ACDE"], "rna"),
    (["ACDE", "AaCDE"], "protein"),
    (["ACDEF", "ACDEF"], "protein"),
    (["AA", "AA"], "protein"),
    (["", ""], "protein"),
])
def test_msa_content_polymer_indels_insertions_and_duplicates_match_uncached(mutant, polymer):
    cache = MsaRowCache()
    cache.extract(encoder, ["ACDE", "AAAA"], "protein")
    for actual, expected in zip(cache.extract(encoder, mutant, polymer), encoder(mutant, polymer)):
        np.testing.assert_array_equal(actual, expected)


def test_msa_rejects_length_mismatch_and_does_not_publish_bad_rows():
    cache = MsaRowCache()
    cache.extract(encoder, ["ACDE"], "protein")
    with pytest.raises(ValueError, match="different row lengths"):
        cache.extract(encoder, ["ACDE", "ABCDE"], "protein")
    with pytest.raises(ValueError, match="invalid residue"):
        cache.extract(encoder, ["ACDE", "AC?E"], "protein")
    assert cache.encoded_rows == 1


def test_msa_cache_is_bounded_and_callers_keep_results_after_eviction():
    cache = MsaRowCache(max_entries=1, max_bytes=100)
    result = cache.extract(encoder, ["AAAA", "CCCC"], "protein")
    assert len(cache._rows) == 1
    assert cache._bytes <= 100
    np.testing.assert_array_equal(result[0], encoder(["AAAA", "CCCC"], "protein")[0])
    cache.clear()
    assert len(cache._rows) == 0


def test_dispatcher_restores_context_on_error_and_isolates_threads():
    module = SimpleNamespace(extract_msa_features=encoder)
    install_msa_row_dispatch(module)
    installed = module.extract_msa_features
    install_msa_row_dispatch(module)
    assert module.extract_msa_features is installed
    caches = [MsaRowCache(), MsaRowCache()]

    def work(cache):
        with msa_row_cache_scope(cache):
            module.extract_msa_features(["AAAA"], "protein")
            module.extract_msa_features(["AAAA"], "protein")
        return cache.encoded_rows, cache.hits

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(work, caches)) == [(1, 1), (1, 1)]
    with pytest.raises(ValueError):
        with msa_row_cache_scope(caches[0]):
            module.extract_msa_features(["?"], "protein")
    module.extract_msa_features(["CCCC"], "protein")
    assert caches[0].encoded_rows == 1


def test_file_asset_loads_once_and_invalidates_in_place_or_replacement(tmp_path):
    path = tmp_path / "ccd.json"
    path.write_text('{"ALA": [1]}')
    calls = []

    def loader(file):
        calls.append(file)
        return json.loads(file.read_text())

    cache = FileObjectCache()
    original = cache.load(path, loader)
    assert cache.load(path, loader) is original
    assert len(calls) == 1
    path.write_text('{"ALA": [2]}')
    assert cache.load(path, loader)["ALA"] == [2]
    replacement = tmp_path / "new.json"
    replacement.write_text('{"ALA": [3]}')
    replacement.replace(path)
    assert cache.load(path, loader)["ALA"] == [3]
    assert len(calls) == 3


def test_failed_asset_load_is_not_published(tmp_path):
    path = tmp_path / "asset"
    path.write_text("bad")
    cache = FileObjectCache()
    with pytest.raises(json.JSONDecodeError):
        cache.load(path, lambda p: json.loads(p.read_text()))
    assert cache._entry is None


def test_copy_on_access_mapping_isolates_mutable_native_ccd_entries():
    source = {"ALA": {"atoms": ["N", "CA"]}, "GLY": {"atoms": ["N"]}}
    request = CopyOnAccessMapping(source)
    assert set(request) == set(source)
    request.get("ALA")["atoms"].append("O")
    request["NEW"] = {"atoms": []}
    del request["GLY"]
    assert source["ALA"]["atoms"] == ["N", "CA"]
    assert "GLY" in source
    assert set(request) == {"ALA", "NEW"}
    assert request.get("GLY") is None
    assert CopyOnAccessMapping(source)["ALA"]["atoms"] == ["N", "CA"]


def test_af3_custom_ccd_is_owner_local_and_clears_identity_based_references():
    shared = {"ALA": {"atom": ("old",)}}

    class Ccd(dict):
        def __init__(self):
            self._dict = shared

        def items(self):
            return self._dict.items()

    components = SimpleNamespace(Ccd=Ccd, cif_dict=SimpleNamespace(parse_multi_data_cif=json.loads))
    features = SimpleNamespace(_REFSTRUCTURE_REFERENCE_CACHE={"old": 1})
    cache = AF3CcdCache()
    custom = '{"ALA": {"atom": ["new"]}}'
    first = cache.get(components, features, custom)
    assert first._dict["ALA"]["atom"] == ("new",)
    assert shared["ALA"]["atom"] == ("old",)
    assert cache.get(components, features, custom) is first
    features._REFSTRUCTURE_REFERENCE_CACHE["new"] = 2
    native = cache.get(components, features, None)
    assert native._dict["ALA"]["atom"] == ("old",)
    assert features._REFSTRUCTURE_REFERENCE_CACHE == {}
    with pytest.raises(ValueError, match="empty"):
        cache.get(components, features, "")


def test_template_content_cache_reuses_only_identical_content_and_returns_copies():
    cache = ContentCache(max_entries=2)
    calls = []

    def parse(text):
        calls.append(text)
        return {"atoms": [text]}

    result = cache.get_or_create("template one", parse)
    result["atoms"].clear()
    assert cache.get_or_create("template one", parse)["atoms"] == ["template one"]
    assert len(calls) == 1
    assert cache.get_or_create("template two", parse)["atoms"] == ["template two"]
    assert len(calls) == 2


def test_native_msa_encoder_delta_parity():
    native = pytest.importorskip("alphafold3.data.msa_features")
    names = pytest.importorskip("alphafold3.constants.mmcif_names")
    cache = MsaRowCache()
    for polymer, rows in [
        (names.PROTEIN_CHAIN, ["ACDE", "AcCDE", "A--E", "AAAA"]),
        (names.RNA_CHAIN, ["ACGU", "AcCGU", "A--U", "AAAA"]),
        (names.DNA_CHAIN, ["ACGT", "AcCGT", "A--T", "AAAA"]),
    ]:
        for query in (rows[0], "CCCC", "GGGG"):
            mutant = [query, *rows[1:]]
            got = cache.extract(native.extract_msa_features, mutant, polymer)
            expected = native.extract_msa_features(mutant, polymer)
            for actual, direct in zip(got, expected):
                np.testing.assert_array_equal(actual, direct)
    with pytest.raises(ValueError, match="Invalid shape"):
        cache.extract(native.extract_msa_features, ["ACDE", "ACDEF"], names.PROTEIN_CHAIN)


def test_native_msa_profile_reports_actual_work():
    pytest.importorskip("alphafold3.data.msa_features")
    from astrolabe.utils.af3_featurization_profile import profile_msa_row_cache

    result = profile_msa_row_cache(sequence_length=16, msa_depth=32, runs=3)
    assert result["bitwise_equal"]
    assert result["encoded_rows_per_mutant"] == [1, 1, 1]
