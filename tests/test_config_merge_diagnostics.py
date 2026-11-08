from omegaconf import OmegaConf
import pytest

from astrolabe.config_merge import merge_configs


@pytest.mark.parametrize("layers", [
    ({"data": {"msa": {}}}, {"data": []}),
    ({"data": {"msa": {}}}, {"other": 1}, {"data": []}),
    ({"data": []}, {"data": {"msa": {}}}),
])
def test_container_mismatch_names_actual_field(layers):
    with pytest.raises(ValueError, match=r"evaluator\.alias\.data: cannot merge"):
        merge_configs(*(OmegaConf.create(layer) for layer in layers), path="evaluator.alias")


def test_legal_overrides_keep_omegaconf_semantics_and_do_not_resolve_unused():
    original = OmegaConf.create({"unused": "${missing.path}", "disabled": {"key": 1}, "values": [1]})
    override = OmegaConf.create({"disabled": None, "values": [2, 3]})
    actual = merge_configs(original, override)
    assert OmegaConf.to_container(actual, resolve=False) == OmegaConf.to_container(
        OmegaConf.merge(original, override), resolve=False,
    )
    assert original.disabled.key == 1
