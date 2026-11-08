"""Process-local equivalent of mask patch01; official vendor files stay intact.

Only caller annotations are restored after the official parser rebuilds atoms.
No energy, state, center, sampling, or model code is replaced.
"""
from __future__ import annotations


def install_design_mask_adapter():
    import numpy as np
    from atomworks.ml.utils.token import get_token_starts, spread_token_wise
    import mpnn.utils.inference as inference

    original = inference.parse_atom_array
    if getattr(original, '_astrolabe_preserve_design_mask', False):
        return False

    def keys(array):
        tokens = array[get_token_starts(array)]
        insertions = (tokens.ins_code if 'ins_code' in tokens.get_annotation_categories()
                      else np.full(tokens.array_length(), '', dtype='<U4'))
        return tokens, list(zip(tokens.chain_id, tokens.res_id, insertions))

    def preserving_parse(array, *args, **kwargs):
        saved = None
        if 'mpnn_designed_residue_mask' in array.get_annotation_categories():
            tokens, source_keys = keys(array)
            if len(set(source_keys)) != len(source_keys):
                raise ValueError('Design mask needs unique (chain_id, res_id, ins_code).')
            saved = dict(zip(source_keys, np.asarray(tokens.mpnn_designed_residue_mask, dtype=bool).tolist()))
        result = original(array, *args, **kwargs)
        if saved is not None and result.get('assemblies'):
            models = result['assemblies']['1']
            parsed = models[0]
            _, parsed_keys = keys(parsed)
            missing = set(parsed_keys) - saved.keys()
            if missing:
                raise ValueError(f'Parsed residues have no input design-mask identity: {sorted(missing)}')
            restored = np.asarray([saved[key] for key in parsed_keys], dtype=bool)
            parsed.set_annotation('mpnn_designed_residue_mask',
                                  spread_token_wise(parsed, restored).astype(bool))
            # AtomArrayStack[0] produces a new AtomArray annotation dictionary.
            # A new annotation on that view does not reach the stack: retain
            # this exact annotated object for the caller's subsequent [0].
            depth = models.stack_depth() if hasattr(models, 'stack_depth') else len(models)
            assemblies = dict(result['assemblies'])
            assemblies['1'] = [parsed] + [models[i] for i in range(1, depth)]
            result = {**result, 'assemblies': assemblies}
        return result

    preserving_parse._astrolabe_preserve_design_mask = True
    preserving_parse.__wrapped__ = original
    inference.parse_atom_array = preserving_parse
    return True
