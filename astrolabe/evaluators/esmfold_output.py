"""Audited ESMFold1 output materialization and CPU PDB finalization.

The model's atom14 -> atom37 operation stays on its original device. Only the
fields consumed by PDB formatting and score publication cross to CPU, once.
Unknown native implementations retain the daemon's original composite path.

Audited sources: Molaison/gh-esm2@43c67d952da30d2471f609848931d20ec8c98532
and Molaison/openfold@8396098268a5675b0ff776ec274f15d380a4ccf5. The precise
source hashes and CPU differential fixture are in tests/fixtures/
esmfold_output_native.json. This module does not import a model or load weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import sys
import textwrap

from astrolabe.evaluators.lifecycle import cpu_tree


SCORE_FIELDS = ('chain_index', 'plddt', 'mean_plddt', 'ptm', 'predicted_aligned_error')
_PDB_FIELDS = ('aatype', 'atom37_atom_exists', 'residue_index', 'plddt', 'chain_index')
_SOURCE_HASHES = {
    'wrapper': 'c8c02d8ac6635864e241d9d3e342d5ef9063cb8bb496de25439c186bf8e5d79c',
    'output_to_pdb': 'e3f7c8a33a0322f7d2dd95aaebefb308c7ab4b229babf067c4c1bca4bba3d698',
    'atom14_to_atom37': 'f7d3e281630fdb0e8338d8d04665ffac7e7c4f14384b6652b54d4b779621a3b8',
    'batched_gather': '0db9097bfa3bb1c854f4a4c5de9c52b966597915fa9d4898f36eb521fe1491da',
    'Protein': 'd92b7c3e111832aa6e985b49e917f1906cb04751b892fb515fb5c11c2d1f55af',
    'to_pdb': 'c6f17231f2556cf355d8ca7b686248d64d2ed22b0569c1280774f1b2a977eab8',
    'get_pdb_headers': 'c086b681057d8cf2d3161891bcd61087eb1e021ef5a0b47a497039d54a12c339',
    '_chain_end': '82f09006cceffbb8b194ddf6f29dafe28b95a4d4f062ec483eaf335941badcaa',
}
_CONSTANTS_HASH = '46281aa1ed454bcbe6a7109792f3bf282c993a39da0d6291fe30a2d64f0b7db8'


def _source_matches(value, name):
    # inspect.getsource follows __wrapped__; an unknown decorator must not be
    # mistaken for the audited function it advertises.
    if hasattr(value, '__wrapped__'):
        return False
    source = '\n'.join(line.rstrip() for line in textwrap.dedent(inspect.getsource(value)).strip().splitlines())
    return hashlib.sha256(source.encode()).hexdigest() == _SOURCE_HASHES[name]


@dataclass(frozen=True)
class _PDBBoundary:
    atom14_to_atom37: object
    protein: object
    to_pdb: object

    def prepare_output(self, output):
        """Call inside the device lease; return CPU tensors with original dtype."""
        positions = self.atom14_to_atom37(output['positions'][-1], output)
        fields = dict.fromkeys((*_PDB_FIELDS, *SCORE_FIELDS))
        # chain_index is optional to the native PDB helper. Do not synthesize it
        # or alter the pinned formatter's existing missing-chain failure.
        selected = {key: output[key] for key in fields
                    if key != 'chain_index' or key in output}
        selected['final_atom_positions'] = positions
        return cpu_tree(selected)

    def finalize_pdb(self, host):
        """Format only materialized host arrays; never move a device tensor."""
        fields = (*_PDB_FIELDS, 'final_atom_positions')
        arrays = {}
        for key in fields:
            if key == 'chain_index' and key not in host:
                continue
            tensor = host[key]
            if tensor.device.type != 'cpu':
                raise ValueError('ESMFold PDB finalization requires CPU tensors')
            arrays[key] = tensor.numpy()
        pdbs = []
        for i in range(arrays['aatype'].shape[0]):
            pred = self.protein(
                aatype=arrays['aatype'][i],
                atom_positions=arrays['final_atom_positions'][i],
                atom_mask=arrays['atom37_atom_exists'][i],
                residue_index=arrays['residue_index'][i] + 1,
                b_factors=arrays['plddt'][i],
                chain_index=arrays['chain_index'][i] if 'chain_index' in arrays else None,
            )
            pdbs.append(self.to_pdb(pred))
        return pdbs


def resolve_pdb_boundary(model):
    """Return the narrow split only for the actual audited native call graph.

    Called once after model construction. Worker source/weights are immutable
    for the session and participate in its identity; hot patching is unsupported.
    A matching name or version string alone never enables this optimization.
    """
    try:
        import numpy as np
        bound = model.output_to_pdb
        if not inspect.ismethod(bound) or bound.__self__ is not model:
            return None
        wrapper = bound.__func__
        native = wrapper.__globals__['output_to_pdb']
        mapper = native.__globals__['atom14_to_atom37']
        protein = native.__globals__['OFProtein']
        formatter = native.__globals__['to_pdb']
        gather = mapper.__globals__['batched_gather']
        symbols = {
            'wrapper': wrapper, 'output_to_pdb': native,
            'atom14_to_atom37': mapper, 'batched_gather': gather,
            'Protein': protein, 'to_pdb': formatter,
            'get_pdb_headers': formatter.__globals__['get_pdb_headers'],
            '_chain_end': formatter.__globals__['_chain_end'],
        }
        if not all(_source_matches(value, name) for name, value in symbols.items()):
            return None
        scope = formatter.__globals__
        constants = scope['residue_constants']
        values = {key: getattr(constants, key) for key in
                  ('atom_types', 'restypes', 'restype_num', 'restype_1to3')}
        digest = hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if (digest != _CONSTANTS_HASH or scope['np'] is not np
                or protein.__post_init__.__globals__['np'] is not np
                or gather.__globals__['torch'] is not sys.modules.get('torch')
                or scope['PDB_CHAIN_IDS'] != 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                or scope['PDB_MAX_CHAINS'] != 62
                or scope['string'].ascii_uppercase != 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            return None
        return _PDBBoundary(mapper, protein, formatter)
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        # This is a compatibility gate, not an attempt to repair vendor code.
        return None
