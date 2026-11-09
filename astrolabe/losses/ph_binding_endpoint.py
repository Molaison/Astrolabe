"""Raw local-His binding endpoint descriptors (benchmark revisions 49/50).

All metrics are (complex - partner_a - partner_b)_P minus the same quantity
in S. No normalization, sign fitting, pH integration, or ensemble is applied.
``local_residues`` is an explicit frozen list of PDB keys (e.g. ``A:35`` or
``A:35B`` with insertion code). It defines BOTH the local packing region and,
by intersection with the current sequence, the His S->P centres. All other His
are neutral native HIS. Partner chain lists are literal structure chain IDs,
not protomer selectors; multi-chain partners are never flattened.

Numerical readouts, without models or relaxation:
    proton_endpoint_from_npz(proton_dir / 'primary', local_residues)
    rosetta_endpoint_from_poses({'S': poses_s, 'P': poses_p}, scorefxn, local_residues)
    read_saved_rosetta_endpoints({'S': paths_s, 'P': paths_p}, local_residues)
Each pose/path mapping has exactly complex, partner_a, partner_b.

Live API:
    ph_binding_endpoint(eval_result, metric, binder_chains, target_chains,
                        local_residues, seed=0, proton_config=None)
Proton config requires python_bin, repo_path, checkpoint_path, hbplus_bin.
The qualified CPU worker is lazy and isolated; no package installs. Rosetta
uses a fresh native process so options and seed resets do not affect the search
process. Rosetta and fa_elec share one local-relax result per content identity.

Protocol sources (ported, not runtime dependencies):
rev49/02_relax.py:set_state/delete_partner and archived rev27/rosetta_ph.py:
_local_rearrange; rev49/03_proton_score.py reference encoding before encoder;
rev50/03_joint_states.py:enumerate_system float64 changed directed entries.
Proton H is F+directedP, NOT Q=F+2P; learned units are not binding free energy.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

SYSTEMS = ("complex", "partner_a", "partner_b")
TOKENS = ("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
          "UNK HIS-P HIS-S HIS-A ASP-P ASP-D ASP-A GLU-P GLU-D GLU-A").split()
AA3 = dict(zip(TOKENS[:20], "ARNDCQEGHILKMFPSTWYV"))
REFERENCE = {"HIS": "HIS-S", "ASP": "ASP-D", "GLU": "GLU-D"}
CHECKPOINT_SHA256 = "a39872250c0b8eb4c0b4cb6472a6edad0d6f6134a50c28b611386e83c592fe7b"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mask(keys):
    _require(not isinstance(keys, (str, bytes)), "local_residues must be an explicit list of chain:residue keys")
    keys = list(keys)
    _require(all(isinstance(k, str) and re.fullmatch(r"[^:\s]+:-?\d+[A-Za-z]?", k) for k in keys),
             "Invalid local_residues key; expected chain:number[insertion_code]")
    _require(len(keys) == len(set(keys)), "Duplicate local_residues keys")
    return tuple(sorted(keys))


def _finite(value):
    value = float(value)
    _require(math.isfinite(value), "Non-finite endpoint score")
    return value


def endpoint_difference(states):
    """Pure strict scalar readout; missing endpoints/systems are errors, not zero."""
    _require(set(states) == {"S", "P"}, "Exactly S and P are required")
    delta = {}
    for state in ("S", "P"):
        _require(set(states[state]) == set(SYSTEMS), "Exactly complex/partner_a/partner_b required")
        c, a, b = (_finite(states[state][s]) for s in SYSTEMS)
        delta[state] = _finite(math.fsum((c, -a, -b)))
    return _finite(delta["P"] - delta["S"])


def proton_table_gap(table, eidx, sequence, rows, token_to_idx, local_residues):
    """Pure audited H(P)-H(S), promoting entries BEFORE subtraction/reduction.

    ``rows`` must contain key and canonical resname in token order. Masks are
    projected into this context; the three-context API checks global coverage.
    """
    keys = _mask(local_residues)
    _require(token_to_idx == {t: i for i, t in enumerate(TOKENS)}, "Not the benchmark V30 encoding")
    table, eidx, sequence = np.asarray(table), np.asarray(eidx), np.asarray(sequence)
    _require(table.dtype == np.float32 and table.ndim == 4, "Expected original float32 [L,K,30,30] table")
    length, neighbors, v1, v2 = table.shape
    _require(length > 0 and neighbors > 0 and v1 == v2 == 30 and len(rows) == length, "Invalid table dimensions")
    _require(eidx.dtype == sequence.dtype == np.int64 and eidx.shape == (length, neighbors)
             and sequence.shape == (length,), "Invalid S/E_idx dtype or shape")
    _require(np.all((sequence >= 0) & (sequence < 30)) and np.all((eidx >= 0) & (eidx < length)), "Table index out of range")
    _require(np.array_equal(eidx[:, 0], np.arange(length))
             and not np.any(eidx[:, 1:] == np.arange(length)[:, None]), "Invalid self slots")
    for begin in range(0, length, 32):
        _require(np.isfinite(table[begin:begin + 32]).all(), "Non-finite Proton table")
    _require(not np.any(table[:, 0, ~np.eye(30, dtype=bool)]), "Self slot is not diagonal")
    by_key = {r["key"]: i for i, r in enumerate(rows)}
    _require(len(by_key) == length, "Duplicate residue keys in token map")
    for i, row in enumerate(rows):
        name = row["resname"]
        _require(name in AA3 and int(sequence[i]) == token_to_idx[REFERENCE.get(name, name)],
                 f"Canonical/background reference mismatch at {row['key']}")
        _require(row.get("token_index", i) == i and row.get("aa", AA3[name]) == AA3[name], "Token identity/order mismatch")
    positions = [by_key[k] for k in keys if k in by_key and rows[by_key[k]]["resname"] == "HIS"]
    current = sequence.copy()
    current[positions] = token_to_idx["HIS-P"]
    ii, kk = np.indices(eidx.shape)
    # Every directed entry once, including incoming edges to selected centres.
    affected = np.isin(ii, positions) | np.isin(eidx, positions)
    ii, kk = ii[affected], kk[affected]
    jj = eidx[ii, kk]
    old = table[ii, kk, sequence[ii], sequence[jj]].astype(np.float64)
    new = table[ii, kk, current[ii], current[jj]].astype(np.float64)
    return _finite((new - old).sum(dtype=np.float64))


def _context_identity(rows_by_system, local_residues):
    identities = [{r["key"]: r["resname"] for r in rows_by_system[s]} for s in SYSTEMS]
    _require(all(len(d) == len(rows_by_system[s]) for d, s in zip(identities, SYSTEMS)), "Duplicate context residue keys")
    c, a, b = identities
    _require(c and a and b and not a.keys() & b.keys() and c == {**a, **b}, "Context identity/partner partition mismatch")
    _require(set(local_residues) <= c.keys(), f"Mask keys absent from complex: {sorted(set(local_residues) - c.keys())}")
    return sorted(k for k in local_residues if c[k] == "HIS")


def proton_endpoint_from_npz(directory, local_residues):
    """Read original benchmark tables; directory contains the three system dirs."""
    keys, rows_by_system, gaps = _mask(local_residues), {}, {}
    for system in SYSTEMS:
        root = Path(directory) / system
        mapping = json.loads((root / "token_map.json").read_text())
        rows_by_system[system] = mapping["residues"]
        with np.load(root / "raw.npz", allow_pickle=False) as raw:
            gaps[system] = proton_table_gap(raw["etab"], raw["E_idx"], raw["S"],
                                          mapping["residues"], mapping["token_to_idx"], keys)
    centres = _context_identity(rows_by_system, keys)
    return {"proton": _finite(math.fsum((gaps["complex"], -gaps["partner_a"], -gaps["partner_b"]))),
            "system_gaps": gaps, "centres": centres}


def rosetta_endpoint_from_poses(poses, scorefxn, local_residues):
    """Read-only numerical scoring of already installed S/P poses; never relaxes.

    Caller supplies ref2015/e_pH=0 scorefxn and PyRosetta poses. This pure readout
    does not install states, reset RNG, or change global options.
    """
    import pyrosetta

    _require(set(poses) == {"S", "P"} and all(set(p) == set(SYSTEMS) for p in poses.values()), "Incomplete saved pose endpoints")
    _require(float(scorefxn.get_weight(pyrosetta.rosetta.core.scoring.e_pH)) == 0.0, "e_pH must be zero")
    keys = _mask(local_residues)
    for state in ("S", "P"):
        rows = {system: _pose_rows(poses[state][system]) for system in SYSTEMS}
        centres = _context_identity(rows, keys)
        for system in SYSTEMS:
            _require(rows[system] == _pose_rows(poses["S"][system]), "S/P residue inventories or sequences differ")
            for i, row in enumerate(rows[system], 1):
                if row["resname"] == "HIS":
                    expected = "HIS_P" if state == "P" and row["key"] in centres else "HIS"
                    _require(poses[state][system].residue(i).type().base_name() == expected,
                             f"Saved endpoint chemistry mismatch at {state}/{system}/{row['key']}")
    total, electrostatic = {}, {}
    for state in ("S", "P"):
        total[state], electrostatic[state] = {}, {}
        for system in SYSTEMS:
            pose = poses[state][system]
            total[state][system] = _finite(scorefxn(pose))
            electrostatic[state][system] = _finite(pose.energies().total_energies()[pyrosetta.rosetta.core.scoring.fa_elec])
    return {"rosetta": endpoint_difference(total), "fa_elec": endpoint_difference(electrostatic),
            "system_scores": {"rosetta": total, "fa_elec": electrostatic}}


def _run_worker(request, python_bin):
    with tempfile.TemporaryDirectory(prefix="astrolabe-ph-endpoint-") as directory:
        root = Path(directory)
        source = root / "request.json"
        source.write_text(json.dumps(request, allow_nan=False))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        # Match the isolated Proton runner, never inherit the host's package path.
        if request["kind"] == "proton":
            env.pop("PYTHONPATH", None)
            env.pop("APPTAINERENV_PYTHONPATH", None)
        proc = subprocess.run([str(python_bin), "-B", str(Path(__file__).resolve()),
                               str(source), str(root / "result.json")],
                              cwd=root, env=env, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"pH endpoint {request['kind']} worker failed:\n{proc.stderr[-12000:]}\n{proc.stdout[-2000:]}")
        return json.loads((root / "result.json").read_text())


def read_saved_rosetta_endpoints(paths, local_residues):
    """Fresh CPU subprocess reads existing poses; no state install or packing."""
    request = {"kind": "rosetta", "local_residues": _mask(local_residues), "paths": {s: {k: str(Path(p).resolve()) for k, p in v.items()}
                                               for s, v in paths.items()}}
    return _run_worker(request, sys.executable)


def _prepare_original(path, binder_chains, target_chains, local_residues, root):
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select

    parser = MMCIFParser(QUIET=True) if Path(path).suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True, PERMISSIVE=False)
    structure = parser.get_structure("original", str(path))
    _require(len(structure) == 1, "Endpoint requires exactly one structural model")
    selected = set(binder_chains) | set(target_chains)
    _require(selected <= {c.id for c in structure[0]}, "Selected chain missing from structure")
    rows = []
    for chain in structure[0]:
        if chain.id not in selected:
            continue
        _require(len(chain.id) == 1 and chain.id.strip(), "PDB endpoint worker requires one-character chain IDs")
        for residue in chain:
            name = residue.resname
            _require(name in AA3 and residue.id[0] == " ", f"Noncanonical selected residue: {chain.id}:{residue.id}")
            _require(all(a in residue for a in ("N", "CA", "C", "O")), "Selected residue lacks backbone atoms")
            for atom in residue.get_unpacked_list():
                _require(not atom.altloc.strip(), "Resolve alternate conformers explicitly before endpoint scoring")
                _require(np.isfinite(atom.coord).all() and atom.occupancy is not None
                         and 0 < atom.occupancy <= 1, "Invalid coordinate/occupancy")
            rows.append({"key": f"{chain.id}:{residue.id[1]}{residue.id[2].strip()}", "resname": name,
                         "aa": AA3[name], "chain_id": chain.id})
    rows_by_system = {s: [r for r in rows if s == "complex" or r["chain_id"] in
                         (binder_chains if s == "partner_a" else target_chains)] for s in SYSTEMS}
    _context_identity(rows_by_system, local_residues)
    paths = {}
    for system in SYSTEMS:
        chains = {r["chain_id"] for r in rows_by_system[system]}
        class KeepChains(Select):
            def accept_chain(self, chain):
                return chain.id in chains
        path_out = root / f"{system}.pdb"
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(path_out), KeepChains())
        paths[system] = str(path_out)
    return paths, rows_by_system


def _pose_rows(pose):
    info = pose.pdb_info()
    rows = [{"key": f"{info.chain(i)}:{info.number(i)}{info.icode(i).strip()}",
             "resname": next((n for n, aa in AA3.items() if aa == pose.residue(i).name1()), "UNK")}
            for i in range(1, pose.total_residue() + 1)]
    _require(len({r["key"] for r in rows}) == len(rows), "Duplicate pose residue keys")
    _require(all(r["resname"] in AA3 for r in rows), "Noncanonical pose identity")
    return rows


def _set_state(pose, positions, state, ros):
    # rev49 wrapper of rev27: idealize new H, then restore ALL common atoms.
    for pos in positions:
        old = pose.residue(pos)
        coords = {old.atom_name(a).strip(): tuple(old.xyz(a)) for a in range(1, old.natoms() + 1)}
        ros.protocols.simple_moves.MutateResidue(pos, state).apply(pose)
        new = pose.residue(pos)
        for a in range(1, new.nheavyatoms() + 1):
            pose.set_xyz(ros.core.id.AtomID(a, pos), ros.numeric.xyzVector_double_t(*coords[new.atom_name(a).strip()]))
        ros.core.conformation.idealize_hydrogens(pose.conformation().residue(pos), pose.conformation())
        for name, xyz in coords.items():
            if new.has(name):
                pose.set_xyz(ros.core.id.AtomID(new.atom_index(name), pos), ros.numeric.xyzVector_double_t(*xyz))


def _local_rearrange(pose, sf, local, ros):
    # Exact archived rev27 protocol used by frozen rev49/50 (no global repack).
    task = ros.core.pack.task.TaskFactory.create_packer_task(pose)
    task.restrict_to_repacking()
    local = set(local)
    for i in range(1, pose.total_residue() + 1):
        rt = task.nonconst_residue_task(i)
        if i not in local:
            rt.prevent_repacking()
            continue
        base = pose.residue(i).type().base_name()
        keep = ros.utility.vector1_std_string()
        keep.append(base)
        rt.restrict_restypes(keep)
        if pose.residue(i).name1() == "H":
            rt.or_fix_his_tautomer(True)
        rt.or_include_current(True)
        rt.or_ex1(True)
        rt.or_ex2(True)
        _require({x.base_name() for x in rt.allowed_residue_types()} == {base}, "Packing changed allowed chemistry")
    if local:
        ros.protocols.minimization_packing.PackRotamersMover(sf, task).apply(pose)
        mm = ros.core.kinematics.MoveMap()
        mm.set_bb(False)
        mm.set_chi(False)
        mm.set_jump(False)
        for i in local:
            mm.set_chi(i, True)
        minimum = ros.protocols.minimization_packing.MinMover(mm, sf, "lbfgs_armijo_nonmonotone", 0.001, True)
        minimum.max_iter(100)
        minimum.apply(pose)


def _rosetta_worker(request):
    import pyrosetta
    ros = pyrosetta.rosetta
    _require(not ros.basic.was_init_called(), "Endpoint worker requires a fresh Rosetta process")
    booleans = {"in:no_fconfig": True, "in:ignore_unrecognized_res": False,
                "run:ignore_zero_occupancy": False, "in:file:load_PDB_components": False,
                "in:missing_density_to_jump": True, "in:detect_disulf": True, "in:obey_ssbond": False,
                "packing:no_optH": True, "packing:flip_HNQ": False, "packing:pack_missing_sidechains": False,
                "run:constant_seed": True, "run:interactive": True,
                "pH:pH_mode": True, "pH:keep_input_protonation_state": True}
    for name in ("beta", "beta_cart", "beta_jan25", "beta_jan25_cart", "beta_nov16", "beta_nov16_cart",
                 "beta_july15", "beta_july15_cart", "gen_potential"):
        booleans[f"corrections:{name}"] = False
    pyrosetta.init("-score:weights ref2015 -multithreading:total_threads 1 "
                   "-multithreading:interaction_graph_threads 1 -mute all "
                   + " ".join(f"-{k} {str(v).lower()}" for k, v in booleans.items()))
    _require(all(ros.basic.options.get_boolean_option(k) == v for k, v in booleans.items()), "Rosetta flag mismatch")
    sf = pyrosetta.create_score_function("ref2015")
    sf.set_weight(ros.core.scoring.e_pH, 0.0)
    if "paths" in request:
        poses = {state: {system: pyrosetta.pose_from_pdb(request["paths"][state][system]) for system in SYSTEMS}
                 for state in ("S", "P")}
        _context_identity({s: _pose_rows(poses["S"][s]) for s in SYSTEMS}, ())
        _require(all(_pose_rows(poses['S'][s]) == _pose_rows(poses['P'][s]) for s in SYSTEMS), "Saved endpoint sequences differ")
        return rosetta_endpoint_from_poses(poses, sf, request["local_residues"])
    root = Path.cwd()
    paths, rows = _prepare_original(request["path"], request["binder_chains"], request["target_chains"], request["local_residues"], root)
    base = pyrosetta.pose_from_pdb(paths["complex"])
    _require(_pose_rows(base) == [{k: r[k] for k in ("key", "resname")} for r in rows["complex"]], "Rosetta import changed identity")
    info = base.pdb_info()
    cysteines = [i for i in range(1, base.total_residue() + 1) if base.residue(i).name1() == "C"]
    for offset, i in enumerate(cysteines):
        for j in cysteines[offset + 1:]:
            if ros.core.conformation.is_disulfide_bond(base.conformation(), i, j):
                _require((info.chain(i) in request['binder_chains']) == (info.chain(j) in request['binder_chains']), "Cross-partner disulfide cannot be separated")
    his = [i for i in range(1, base.total_residue() + 1) if base.residue(i).name1() == "H"]
    _set_state(base, his, "HIS", ros)
    keys = [r["key"] for r in rows["complex"]]
    centres = [i for i in his if keys[i - 1] in request["local_residues"]]
    poses = {}
    for state in ("S", "P"):
        bound = base.clone()
        _set_state(bound, centres, "HIS_P" if state == "P" else "HIS", ros)
        poses[state] = {"complex": bound}
        for system, keep in (("partner_a", request["binder_chains"]), ("partner_b", request["target_chains"])):
            free = bound.clone()  # independent free starts from RAW endpoint bound
            blocks = []
            for i in range(1, free.total_residue() + 1):
                if free.pdb_info().chain(i) not in keep:
                    if blocks and i == blocks[-1][1] + 1:
                        blocks[-1][1] = i
                    else:
                        blocks.append([i, i])
            for first, last in reversed(blocks):
                free.delete_residue_range_slow(first, last)
            poses[state][system] = free
        for system, pose in poses[state].items():
            original_rows = _pose_rows(pose)
            local = [i + 1 for i, r in enumerate(original_rows) if r["key"] in request["local_residues"]]
            frozen = {(i, a): tuple(pose.residue(i).xyz(a)) for i in range(1, pose.total_residue() + 1)
                      for a in range(1, pose.residue(i).natoms() + 1)
                      if i not in local or pose.residue(i).atom_is_backbone(a)}
            types = [pose.residue(i).name() for i in range(1, pose.total_residue() + 1)]
            tree = str(pose.fold_tree())
            ros.numeric.random.rg().set_seed(request["seed"])
            for _ in range(2):
                _local_rearrange(pose, sf, local, ros)
                _require(types == [pose.residue(i).name() for i in range(1, pose.total_residue() + 1)]
                         and original_rows == _pose_rows(pose) and tree == str(pose.fold_tree()), "Relaxation changed identity/chemistry/fold tree")
                _require(all(math.dist(xyz, tuple(pose.residue(i).xyz(a))) <= 1e-6 for (i, a), xyz in frozen.items()), "Relaxation moved fixed atoms")
    result = rosetta_endpoint_from_poses(poses, sf, request["local_residues"])
    result["centres"] = [keys[i - 1] for i in centres]
    return result


def _proton_worker(request):
    cfg = request["proton_config"]
    repo = Path(cfg["repo_path"])
    sys.path[:0] = [str(repo / "foundry/src"), str(repo / "foundry/models/mpnn/src")]
    os.environ["HBPLUS_PATH"] = str(cfg["hbplus_bin"])
    os.environ["EV6_HIS_PROB_THR"] = "0.30"
    os.environ["EV6_ACID_PROB_THR"] = "0.06"
    import random
    import torch
    from atomworks.ml.utils.token import get_token_starts
    from mpnn.potts_inference import load_model, prepare_potts_input, _resolve_device
    from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_V6_TOKEN_ENCODING

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _require(str(_resolve_device()) == "cpu", "Endpoint Proton must run on CPU")
    checkpoint = Path(cfg["checkpoint_path"])
    _require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256, "Not the frozen benchmark checkpoint")
    paths, rows = _prepare_original(request["path"], request["binder_chains"], request["target_chains"], request["local_residues"], Path.cwd())
    model = load_model(checkpoint, extended_vocab="v6")
    encoding = POTTS_MPNN_V6_TOKEN_ENCODING.token_to_idx
    _require(encoding == {t: i for i, t in enumerate(TOKENS)}, "Unexpected Proton V30 encoding")
    gaps = {}
    for system in SYSTEMS:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        batch = prepare_potts_input(paths[system], extended_vocab="v6", structure_noise=0.0,
                                    deterministic=True, protonation_seed=0)
        features = batch["network_input"]["input_features"]
        tokens = batch["atom_array"][get_token_starts(batch["atom_array"])]
        processed = [{"key": f"{a.chain_id}:{a.res_id}{str(a.ins_code).strip()}", "resname": str(a.res_name)} for a in tokens]
        expected = [{k: r[k] for k in ("key", "resname")} for r in rows[system]]
        _require(processed == expected and features["S"].shape == (1, len(expected))
                 and bool(features["residue_mask"].all()), "Proton preparation changed residue identity/order/mask")
        native = features["S"][0].detach().cpu().numpy()
        _require([TOKENS[int(t)].split("-")[0] for t in native] == [r['resname'] for r in expected], "Native Proton canonical sequence changed")
        sequence = torch.tensor([encoding[REFERENCE.get(r["resname"], r["resname"])] for r in processed], dtype=torch.long)
        features["S"] = sequence[None].clone()  # canonical background BEFORE encoder
        with torch.inference_mode():
            table, eidx = model.run_potts_encoder(features)
        _require(torch.equal(features['S'], sequence[None]), "Encoder changed reference tokens")
        gaps[system] = proton_table_gap(table[0].cpu().numpy(), eidx[0].cpu().numpy(), sequence.numpy(),
                                        processed, encoding, request["local_residues"])
    centres = _context_identity(rows, request["local_residues"])
    return {"proton": _finite(math.fsum((gaps['complex'], -gaps['partner_a'], -gaps['partner_b']))),
            "system_gaps": gaps, "centres": centres}


@lru_cache(maxsize=8)
def _live_cached(request_json, source_sha256):
    request = json.loads(request_json)
    _require(hashlib.sha256(Path(request['path']).read_bytes()).hexdigest() == source_sha256, "Source changed before scoring")
    python_bin = request['proton_config']['python_bin'] if request['kind'] == 'proton' else sys.executable
    result = _run_worker(request, python_bin)
    _require(hashlib.sha256(Path(request['path']).read_bytes()).hexdigest() == source_sha256, "Source changed while scoring")
    return json.dumps(result, allow_nan=False)



if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    result = _proton_worker(request) if request["kind"] == "proton" else _rosetta_worker(request)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
    raise SystemExit(0)

from astrolabe.losses.base import loss_registry


@loss_registry.register(
    name="ph_binding_endpoint",
    description="Raw local-His endpoint binding difference P-S (Proton H, ref2015, or fa_elec)",
    required_params=["metric", "binder_chains", "target_chains", "local_residues"],
    optional_params=["seed", "proton_config"],
    supports_oligomer_specific=True,
)
def ph_binding_endpoint(eval_result, metric, binder_chains, target_chains, local_residues,
                        seed=0, proton_config=None):
    """Native raw producer; return {oligomer_name: finite scalar} for LossEngine."""
    _require(metric in {"proton", "rosetta", "fa_elec"}, "metric must be proton, rosetta, or fa_elec")
    _require(not isinstance(binder_chains, str) and not isinstance(target_chains, str), "Use explicit chain lists")
    binder, target = list(binder_chains), list(target_chains)
    _require(binder and target and all(isinstance(c, str) and len(c) == 1 and c.strip() for c in binder + target)
             and len(set(binder + target)) == len(binder + target), "Partner chains must be nonempty, unique and disjoint")
    keys = _mask(local_residues)
    _require(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**31, "seed must be an integer in [0, 2**31)")
    cfg = None
    if metric == 'proton':
        _require(proton_config is not None, "proton_config is required for live Proton")
        cfg = dict(proton_config)
        _require(set(cfg) == {'python_bin', 'repo_path', 'checkpoint_path', 'hbplus_bin'}, "proton_config requires exactly python_bin/repo_path/checkpoint_path/hbplus_bin")
        cfg = {k: str(Path(v).expanduser().resolve(strict=True)) for k, v in cfg.items()}
    result = {}
    for name in eval_result.names:
        path = Path(eval_result.complex_paths[name]).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        request = dict(kind='proton' if metric == 'proton' else 'rosetta', path=str(path),
                       binder_chains=binder, target_chains=target, local_residues=keys,
                       seed=seed, proton_config=cfg)
        request_json = json.dumps(request, sort_keys=True, allow_nan=False)
        values = json.loads(_live_cached(request_json, digest))
        result[name] = _finite(values[metric])
    return result

