"""Isolated native RFD4 generation and immutable, content-addressed results."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading

from astrolabe.mutators.rfd4_types import GeneratedStructure, GenerationRequest

_PROTOCOL = "astrolabe-rfd4-native-v1"
_WORKER = Path(__file__).parent / "wrappers" / "rfd4_worker.py"
_REPO = Path(__file__).resolve().parents[2]
_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_request(python_bin, source_dir, operation, payload, *, env, timeout=1800):
    result = subprocess.run([str(python_bin), str(_WORKER), operation],
                            input=_json(payload), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, cwd=source_dir, timeout=timeout)
    lines = result.stdout.decode(errors="replace").splitlines()
    replies = [line.removeprefix("RFD4_RESULT=") for line in lines
               if line.startswith("RFD4_RESULT=")]
    if result.returncode or len(replies) != 1:
        raise RuntimeError(f"RFD4 CPU {operation} failed ({result.returncode}): "
                           f"{result.stderr.decode(errors='replace')[-6000:]}")
    return json.loads(replies[0])


def native_interaction_metrics(structure_path, *, python_bin, source_dir):
    """Read native interaction counts on CPU, with no model, cache or fingerprint."""
    source_dir = Path(source_dir).resolve()
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join((str(_REPO), str(source_dir / "src"), str(source_dir))),
           "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
           "TOKENIZERS_PARALLELISM": "false", "CLUSTER": "local", "PROJECT_ROOT": str(source_dir)}
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONUSERBASE", None)
    return _cpu_request(Path(python_bin).absolute(), source_dir, "interaction_metrics",
                        {"structure_path": str(Path(structure_path).resolve())}, env=env)


def validate_generation(generation=None):
    """Resolve the deliberately small supported native inference interface."""
    if generation is not None and not isinstance(generation, Mapping):
        raise ValueError("generation must be a mapping")
    supplied = dict(generation or {})
    allowed = {"mode", "nsteps", "partial_t_backbone", "partial_t_latent",
               "partial_trajectory_fraction_steps", "recycle_strategy", "precision"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unknown RFD4 generation settings: {sorted(unknown)}")
    mode = supplied.pop("mode", "partial")
    if mode not in ("partial", "inpaint"):
        raise ValueError("generation mode must be partial or inpaint")
    if mode == "inpaint" and {"partial_t_backbone", "partial_t_latent",
                              "partial_trajectory_fraction_steps"} & supplied.keys():
        raise ValueError("inpaint mode does not accept partial diffusion settings")
    values = {"nsteps": 60, "recycle_strategy": "medium/front_loaded",
              "precision": "bf16-mixed", **supplied}
    fraction = "partial_trajectory_fraction_steps" in supplied
    if fraction and {"partial_t_backbone", "partial_t_latent"} & supplied.keys():
        raise ValueError("Trajectory fraction and modality start times are mutually exclusive")
    if mode == "inpaint":
        # Missing partial args become None in the worker's native full-noise config.
        values["mode"] = mode
    elif not fraction:
        values.setdefault("partial_t_backbone", 0.8)
        values.setdefault("partial_t_latent", 0.8)
    if type(values["nsteps"]) is not int or values["nsteps"] < 1:
        raise ValueError("nsteps must be a positive integer")
    # Native preset bands cover exactly 60 intervals. Do not invent a rescaling rule.
    if values["nsteps"] != 60:
        raise ValueError("Native recycle presets require nsteps=60; schedule rescaling is unsupported")
    strategies = {f"{level}/{shape}" for level in ("low", "medium", "high")
                  for shape in ("flat", "front_loaded", "midpeak")}
    strategies.update({"thinking_low", "thinking_medium", "thinking_high"})
    if values["recycle_strategy"] not in strategies:
        raise ValueError("Unsupported native recycle_strategy")
    if values["precision"] not in {"bf16-mixed", "16-mixed", "32-true"}:
        raise ValueError("precision must be bf16-mixed, 16-mixed, or 32-true")
    for key in ("partial_t_backbone", "partial_t_latent", "partial_trajectory_fraction_steps"):
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        upper_ok = value <= 1 if key.endswith("fraction_steps") else value < 1
        if not 0 < value or not upper_ok:
            raise ValueError(f"Invalid native partial diffusion value for {key}")
        if key.endswith("fraction_steps") and round(value * values["nsteps"]) < 1:
            raise ValueError("Trajectory fraction must select at least one native interval")
    return values


class RFD4Backend:
    def __init__(self, *, python_bin, source_dir, model_dir, cache_dir, generation=None,
                 max_fixed_atom_rmsd=0.5, request_timeout=1800):
        self.python_bin = str(Path(python_bin).absolute())
        self.source_dir = Path(source_dir).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.generation = validate_generation(generation)
        self.max_fixed_atom_rmsd = float(max_fixed_atom_rmsd)
        self.request_timeout = float(request_timeout)
        if not math.isfinite(self.max_fixed_atom_rmsd) or self.max_fixed_atom_rmsd < 0:
            raise ValueError("max_fixed_atom_rmsd must be finite and nonnegative")
        if not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        if not Path(self.python_bin).is_file():
            raise FileNotFoundError(self.python_bin)
        for path in (self.source_dir / "src/rfproteina/generate.py",
                     self.source_dir / "configs/inference.yaml",
                     self.model_dir / "model.safetensors", self.model_dir / "config.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._client = None
        self._client_identity = None
        self._digests = {}
        self._mutex = threading.RLock()

    def _env(self):
        triton_cache = self.cache_dir / ".compiler" / "triton"
        inductor_cache = self.cache_dir / ".compiler" / "torchinductor"
        for path in (triton_cache, inductor_cache):
            path.mkdir(parents=True, exist_ok=True)
        return {"PYTHONPATH": os.pathsep.join((str(_REPO), str(self.source_dir / "src"),
                                               str(self.source_dir))),
                "TRITON_CACHE_DIR": str(triton_cache),
                "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONNOUSERSITE": "1", "PYTHONHOME": "", "PYTHONUSERBASE": "",
                "CLUSTER": "local", "PROJECT_ROOT": str(self.source_dir)}

    def _cpu(self, operation, payload):
        env = {**os.environ, **self._env(), "CUDA_VISIBLE_DEVICES": ""}
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONUSERBASE", None)
        return _cpu_request(self.python_bin, self.source_dir, operation, payload,
                            env=env, timeout=self.request_timeout)

    def _identity_file(self, path):
        path = Path(path)
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        old = self._digests.get(str(path))
        if old is None or old[0] != stamp:
            digest = _sha(path)
            after = path.stat()
            after_stamp = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            if after_stamp != stamp:
                raise RuntimeError(f"Scientific asset changed while hashing: {path}")
            self._digests[str(path)] = (stamp, digest)
        return self._digests[str(path)][1]

    def fingerprint(self):
        """Resolve actual assets and CPU-only environment identity before Runtime caching."""
        with self._mutex:
            probe = self._cpu("identity", {"source_dir": str(self.source_dir),
                                           "model_dir": str(self.model_dir),
                                           "generation": self.generation})
            dependency_files = probe.pop("dependency_files")
            dependencies = {}
            for path in dependency_files:
                stat = Path(path).stat()
                dependencies[path] = [stat.st_dev, stat.st_ino, stat.st_size,
                                      stat.st_mtime_ns, stat.st_ctime_ns]
            dependency_records = {path: self._identity_file(path) for path in dependency_files
                                  if Path(path).name in {"METADATA", "RECORD", "direct_url.json"}}
            source = {}
            for root in (self.source_dir / "src", self.source_dir / "configs"):
                for path in sorted(root.rglob("*")):
                    if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json", ".so"}:
                        source[str(path.relative_to(self.source_dir))] = self._identity_file(path)
            for name in ("pyproject.toml", ".project-root", ".env", ".env.local"):
                path = self.source_dir / name
                if path.is_file():
                    source[name] = self._identity_file(path)
            model = {str(path.relative_to(self.model_dir)): self._identity_file(path)
                     for path in sorted(self.model_dir.rglob("*")) if path.is_file()}
            adapter = {str(path.relative_to(_REPO)): self._identity_file(path) for path in
                       (_WORKER, Path(__file__), _REPO / "astrolabe/mutators/rfd4_types.py",
                        _REPO / "astrolabe/model_execution.py", _REPO / "astrolabe/model_daemon.py",
                        *sorted(_WORKER.parent.glob("rfd4_*worker.py")))}
            device = {key: os.environ.get(key) for key in
                      ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
                       "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                       "DISABLE_CUEQUIVARIANCE", "CUEQ_DISABLE_AOT_TUNING", "CUEQ_DEFAULT_CONFIG",
                       "NAN_CHECK", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap",
                                         "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
                device["inventory"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
            except (OSError, subprocess.TimeoutExpired):
                device["inventory"] = "unavailable"
            revision = subprocess.run(["git", "-C", str(self.source_dir), "rev-parse", "HEAD"],
                                      capture_output=True, text=True, timeout=15)
            return {"protocol": _PROTOCOL, "backend": "native", "device": device,
                    "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                    "source_dir": str(self.source_dir), "source": source,
                    "model_dir": str(self.model_dir), "model": model,
                    "adapter": adapter, "interpreter": self._identity_file(self.python_bin),
                    "environment": probe, "dependencies": dependencies,
                    "dependency_records": dependency_records,
                    "dependency_identity": "installed_records_plus_device_inode_size_mtime_ns_ctime_ns",
                    "generation": dict(self.generation),
                    "max_fixed_atom_rmsd_angstrom": self.max_fixed_atom_rmsd,
                    "fixed_atom_gate": "engineering_tolerance_not_biologically_validated",
                    "structure_status": "generated_unrefolded"}

    def _request(self, request):
        if type(request.seed) is not int or not 0 <= request.seed < 2**32:
            raise ValueError("RFD4 seed must be an integer in [0, 2**32)")
        chains = request.chains
        if not chains or len({c.chain for c in chains}) != len(chains):
            raise ValueError("Physical chains must be nonempty and unique")
        for chain in chains:
            if not chain.chain or not chain.protomer or not isinstance(chain.sequence, str) or not chain.sequence:
                raise ValueError("Physical chains require a nonempty chain, protomer and sequence/CCD")
            if chain.kind not in {"protein", "ligand"}:
                raise ValueError("Chain kind must be protein or ligand")
            if chain.kind == "protein" and set(chain.sequence) - _AA:
                raise ValueError("Only canonical protein chain sequences are supported")
            if chain.kind == "ligand" and not re.fullmatch(r"[A-Z0-9][A-Z0-9_:+.\-]*", chain.sequence):
                raise ValueError("A ligand sequence must be one explicit uppercase CCD identifier")
        valid = {(c.chain, i) for c in chains for i in range(1, c.residue_count + 1)}
        if any(type(r.position) is not int for r in
               (*request.plan.released, *request.plan.fixed_coordinates)):
            raise ValueError("Residue ordinals must be integers")
        released = {(r.chain, r.position) for r in request.plan.released}
        fixed = {(r.chain, r.position) for r in request.plan.fixed_coordinates}
        if not released <= valid or not fixed <= valid:
            raise ValueError("Plan references must be one-based ordinals within physical chains")
        ligand_chains = {c.chain for c in chains if c.kind == "ligand"}
        reconstruction = request.plan.reconstruction
        if reconstruction is not None:
            from .rfd4_reconstruction import reconstruction_bounds
            if set(reconstruction) != {"chain", "length_min", "length_max", "motifs"}:
                raise ValueError("Invalid variable reconstruction request fields")
            lower, upper = reconstruction_bounds({k: reconstruction[k] for k in ("length_min", "length_max")})
            proteins = [c for c in chains if c.kind == "protein"]
            if (self.generation.get("mode") != "inpaint" or len(proteins) != 1
                    or not ligand_chains or reconstruction["chain"] != proteins[0].chain
                    or not lower <= len(proteins[0].sequence) <= upper):
                raise ValueError("Variable reconstruction requires bounded full-noise binder and fixed ligands")
            if (released != {(proteins[0].chain, i) for i in range(1, proteins[0].residue_count + 1)}
                    or fixed != {(c, 1) for c in ligand_chains}):
                raise ValueError("Reconstruction must release the whole scaffold and fix only ligand residues")
            motifs = reconstruction["motifs"]
            ids, sites = set(), set()
            if not isinstance(motifs, (tuple, list)) or len(motifs) >= lower:
                raise ValueError("Invalid motif count for reconstruction bounds")
            for motif in motifs:
                if set(motif) != {"id", "source", "atoms"}:
                    raise ValueError("Invalid motif record fields")
                site = (motif["source"]["chain"], motif["source"]["position"])
                names = motif["atoms"]
                if (not isinstance(motif["id"], str) or not motif["id"] or motif["id"] in ids
                        or site not in released or site in sites or type(site[1]) is not int
                        or not names or len(set(names)) != len(names)
                        or any(not isinstance(n, str) or not n.strip() or n == "OXT" for n in names)):
                    raise ValueError("Motifs require unique source sites, IDs and represented atom names")
                ids.add(motif["id"])
                sites.add(site)
        if any(chain in ligand_chains for chain, _ in released):
            raise ValueError("Ligand residues must never be released")
        if released & fixed:
            raise ValueError("Released residues cannot have all-heavy fixed coordinates")
        if len(released) != len(request.plan.released) or len(fixed) != len(request.plan.fixed_coordinates):
            raise ValueError("Duplicate residue references in condition plan")
        stable = fixed | {(chain, 1) for chain in ligand_chains}
        for hbond in request.plan.hbonds:
            site = (hbond.atom.chain, hbond.atom.position)
            if type(hbond.atom.position) is not int or site not in stable or site in released or hbond.role not in {"DONOR", "ACCEPTOR", "BOTH"}:
                raise ValueError("C_HB requires a stable atom with fixed identity and coordinates and its own role")
        if set(request.plan.excluded_amino_acids) - _AA or len(set(request.plan.excluded_amino_acids)) == 20:
            raise ValueError("Invalid forbidden canonical amino acids")
        for chain in chains:
            tied = [c for c in chains if c.protomer == chain.protomer]
            if len(tied) > 1 and any((c.kind, c.sequence) != (chain.kind, chain.sequence) for c in tied):
                raise ValueError("Copies of one protomer disagree in kind or sequence/CCD")
            if len(tied) > 1 and any(c == chain.chain for c, _ in released):
                raise ValueError("Tied mutable physical copies are not supported natively")
        feedback = json.loads(request.plan.feedback_json)
        _json(feedback)
        path = Path(request.structure_path)
        suffix = path.suffix.lower()
        if suffix not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError("RFD4 input must be uncompressed PDB or CIF")
        data = path.read_bytes()
        record = asdict(request)
        record.pop("structure_path")
        record.update(source_file="source" + suffix, source_sha256=hashlib.sha256(data).hexdigest())
        return json.loads(_json(record)), data

    def _read(self, root, record, identity, digest, *, cache_hit):
        try:
            manifest = json.loads((root / "manifest.json").read_text())
            if (manifest["request_digest"] != digest or manifest["request"] != record
                    or manifest["fingerprint"] != identity
                    or manifest["structure_status"] != "generated_unrefolded"):
                raise ValueError("Cache scientific identity mismatch")
            expected = {record["source_file"], "request.json", "annotated_input.cif", "generated.cif"}
            if set(manifest["checksums"]) != expected:
                raise ValueError("Incomplete cache artifact inventory")
            for name, checksum in manifest["checksums"].items():
                if _sha(root / name) != checksum:
                    raise ValueError(f"Cache checksum mismatch: {name}")
            if _sha(root / record["source_file"]) != record["source_sha256"]:
                raise ValueError("Cache source content identity mismatch")
            if json.loads((root / "request.json").read_text()) != record:
                raise ValueError("Cache request mismatch")
            validation = self._cpu("validate", {"root": str(root), "request": record,
                                                "max_fixed_atom_rmsd": self.max_fixed_atom_rmsd})
            if validation != manifest["validation"]:
                raise ValueError("Cache semantic validation mismatch")
            return GeneratedStructure(str(root / "generated.cif"), str(root / "annotated_input.cif"),
                                      str(root / "manifest.json"),
                                      tuple(tuple(x) for x in validation["sequences"]), digest,
                                      manifest["checksums"]["generated.cif"], cache_hit)
        except Exception as error:
            raise RuntimeError(f"Invalid RFD4 cache entry preserved at {root}: {error}") from error

    def generate(self, request: GenerationRequest) -> GeneratedStructure:
        record, source = self._request(request)
        with self._mutex:
            identity = self.fingerprint()
            digest = hashlib.sha256(_json({"request": record, "fingerprint": identity})).hexdigest()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            root = self.cache_dir / digest
            with (self.cache_dir / (digest + ".lock")).open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if root.exists():
                    return self._read(root, record, identity, digest, cache_hit=True)
                staging = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=self.cache_dir))
                (staging / record["source_file"]).write_bytes(source)
                (staging / "request.json").write_bytes(_json(record))
                prepared = self._cpu("prepare", {"root": str(staging), "request": record})
                identity_digest = hashlib.sha256(_json(identity)).hexdigest()
                if self._client is not None and self._client_identity != identity_digest:
                    raise RuntimeError("RFD4 assets changed under a resident model; shut it down before reuse")
                if self._client is None:
                    from astrolabe.model_execution import NativeModelClient, register_model_owner
                    self._client = NativeModelClient(
                        "rfd4", "astrolabe.mutators.wrappers.rfd4_worker:create_handler",
                        {"source_dir": str(self.source_dir), "model_dir": str(self.model_dir),
                         "generation": self.generation}, python_bin=self.python_bin,
                        cwd=str(self.source_dir), env=self._env(), request_timeout=self.request_timeout,
                        startup_timeout=self.request_timeout)
                    self._client_identity = identity_digest
                    register_model_owner(self)
                result = self._client.request({"root": str(staging), "request": record})
                generated = Path(result["structure_path"])
                if not generated.is_relative_to(staging) or not generated.is_file():
                    raise RuntimeError("Native output is outside its private request directory")
                (staging / "generated.cif").write_bytes(generated.read_bytes())
                validation = self._cpu("validate", {"root": str(staging), "request": record,
                                                    "max_fixed_atom_rmsd": self.max_fixed_atom_rmsd})
                if prepared["mapping"] != validation["mapping"]:
                    raise RuntimeError("Input mapping changed during generation")
                if self.fingerprint() != identity:
                    raise RuntimeError("Scientific assets changed during native generation")
                checksums = {name: _sha(staging / name) for name in
                             (record["source_file"], "request.json", "annotated_input.cif", "generated.cif")}
                manifest = {"request_digest": digest, "request": record, "fingerprint": identity,
                            "structure_status": "generated_unrefolded", "validation": validation,
                            "checksums": checksums, "native_run": result.get("provenance", {})}
                (staging / "manifest.json").write_bytes(_json(manifest))
                for path in staging.rglob("*"):
                    if path.is_file():
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                for path in [p for p in staging.rglob("*") if p.is_dir()] + [staging]:
                    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                os.rename(staging, root)
                fd = os.open(self.cache_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return GeneratedStructure(str(root / "generated.cif"), str(root / "annotated_input.cif"),
                                          str(root / "manifest.json"),
                                          tuple(tuple(x) for x in validation["sequences"]), digest,
                                          checksums["generated.cif"], False)

    def shutdown(self):
        with self._mutex:
            if self._client is not None:
                self._client.shutdown()
                self._client = None
                self._client_identity = None
