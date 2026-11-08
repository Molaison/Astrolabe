"""Retained generated coordinates, separate from independent evaluator evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

from astrolabe.mutators.rfd4_types import ChainSpec, GenerationResult

from .legacy import LegacyFileBytesCodec, _decode_chemical, resolve_legacy_index_map
from .store import DataclassCodec
from .types import Artifact, Blob, Chemical, Dependency, Key, Polymer, Sample, key_for, manifest


@dataclass(frozen=True)
class GeneratedStructureRecord:
    base_candidate: Key
    proposed_candidate: Key
    sample: Sample
    assembly: str
    chains: tuple[ChainSpec, ...]
    sequences: tuple[tuple[str, str], ...]
    request_digest: str
    structure_digest: str
    generated_cif: Blob
    annotated_input: Blob
    metadata: Blob
    validation: str = "generated_unrefolded"


GENERATED_STRUCTURE_CODEC = DataclassCodec(
    GeneratedStructureRecord, "astrolabe.design.generated-structure/v1",
    nested_types=(ChainSpec,),
)
_BYTES = LegacyFileBytesCodec()
GENERATED_STRUCTURE_REFERENCE_CODECS = (GENERATED_STRUCTURE_CODEC, _BYTES)


def _validate_sequences(chains, sequences, occurrences, candidate, data, *, source=False):
    expected = {chain.chain: chain.sequence for chain in chains} if source else dict(sequences)
    if (len(expected) != len(chains) or len(sequences) != len(chains)
            or {chain.chain for chain in chains} != set(expected)):
        raise ValueError("Generated structure chain mapping is incomplete or duplicated")
    assignments = {assignment.variable: assignment.value for assignment in candidate.assignments}
    chain_to_proto = {chain.chain: chain.protomer for chain in chains}
    chain_specs = {chain.chain: chain for chain in chains}
    pieces = {chain: [] for chain in expected}
    for occurrence in occurrences:
        if occurrence.chain not in pieces:
            raise ValueError("Generated structure occurrence references an unexpected chain")
        index = resolve_legacy_index_map(occurrence, candidate, data)
        value = assignments.get(occurrence.variable)
        spec = chain_specs[occurrence.chain]
        if isinstance(value, Chemical):
            chemical = _decode_chemical(data, value.graph)
            if (spec.kind != "ligand" or chemical["kind"] != "ccd"
                    or not chemical.get("is_fixed", False)
                    or chemical["value"] != spec.sequence
                    or expected[occurrence.chain] != spec.sequence):
                raise ValueError("Generated structure ligand must retain its fixed CCD identity and assignment")
            length, tokens = 1, chemical["value"]
        elif (isinstance(value, Polymer) and value.kind == "protein"
                and not value.modifications and spec.kind == "protein"
                and all(token in frozenset("ACDEFGHIKLMNPQRSTVWY") for token in value.tokens)):
            length, tokens = len(value.tokens), "".join(value.tokens)
        else:
            raise ValueError("Generated structure must bind canonical protein or fixed CCD assignments")
        if (index["protomer"] != chain_to_proto[occurrence.chain]
                or index["variable"] != occurrence.variable
                or occurrence.component != occurrence.variable):
            raise ValueError("Generated structure protomer/variable occurrence mismatch")
        if index["assembly"] != occurrence.assembly or index["chain"] != occurrence.chain:
            raise ValueError("Generated structure physical occurrence mismatch")
        start, stop = index["protomer_range"]
        if index["variable_range"] != [0, length] or stop - start != length:
            raise ValueError("Generated structure occurrence has an unsupported index mapping")
        pieces[occurrence.chain].append((start, stop, tokens))
    for chain, spans in pieces.items():
        cursor, sequence = 0, ""
        for start, stop, tokens in sorted(spans):
            if start != cursor:
                raise ValueError("Generated structure occurrences overlap or leave gaps")
            cursor, sequence = stop, sequence + tokens
        if sequence != expected[chain]:
            raise ValueError("Generated structure sequence differs from candidate assignments")


def _read_files(result):
    from astrolabe.mutators.rfd4 import structure_sequences

    generated = result.generated
    structure = Path(generated.structure_path).read_bytes()
    annotated = Path(generated.annotated_input_path).read_bytes()
    metadata = Path(generated.manifest_path).read_bytes()
    if hashlib.sha256(structure).hexdigest() != generated.structure_digest:
        raise ValueError("Generated structure digest does not match retained bytes")
    if structure_sequences(generated.structure_path, result.chains) != dict(generated.sequences):
        raise ValueError("Generated structure coordinates disagree with reported sequences")
    json.loads(metadata)
    return structure, annotated, metadata


def bind_generated_structure(result, *, data, system, base, proposed, sample, inputs):
    if not isinstance(result, GenerationResult):
        raise TypeError("Expected a typed GenerationResult")
    if base.system != system.identity or proposed.system != system.identity:
        raise ValueError("Generated structure belongs to another System")
    if system.edit(base, proposed.assignments, data=data) != proposed:
        raise ValueError("Generated structure candidate identity differs from its assignments")
    occurrences = tuple(occurrence for occurrence in system.occurrences
                        if occurrence.assembly == result.assembly)
    _validate_sequences(result.chains, result.generated.sequences, occurrences, base, data, source=True)
    _validate_sequences(result.chains, result.generated.sequences, occurrences, proposed, data)
    structure, annotated, metadata = _read_files(result)
    record = GeneratedStructureRecord(
        base.identity, proposed.identity, sample, result.assembly, result.chains,
        result.generated.sequences, result.generated.request_digest,
        result.generated.structure_digest, data.put(structure, _BYTES),
        data.put(annotated, _BYTES), data.put(metadata, _BYTES),
    )
    content = data.put(record, GENERATED_STRUCTURE_CODEC)
    artifact_inputs = manifest(inputs.dependencies + (
        Dependency("generated-structure", content.identity),
        Dependency("generated-request", key_for(result.generated.request_digest)),
    ))
    return Artifact(
        identity=key_for(("artifact/generated-structure/v1", base.identity,
                          proposed.identity, occurrences, sample, content, artifact_inputs)),
        candidate=proposed.identity, occurrences=occurrences, sample=sample,
        content=content, derived_from=(base.identity,), inputs=artifact_inputs,
    )


def read_generated_structures(proposal, data) -> tuple[GeneratedStructureRecord, ...]:
    """Read retained proposals after restart without importing the model backend."""
    for codec in GENERATED_STRUCTURE_REFERENCE_CODECS:
        data.register(codec)
    records = []
    for artifact in proposal.artifacts:
        if artifact.content.codec != GENERATED_STRUCTURE_CODEC.identity:
            continue
        record = data.get(artifact.content, GENERATED_STRUCTURE_CODEC)
        if (record.base_candidate != proposal.base
                or record.proposed_candidate != proposal.candidate.identity
                or artifact.candidate != proposal.candidate.identity
                or record.sample != artifact.sample
                or record.validation != "generated_unrefolded"):
            raise ValueError("Generated structure candidate/sample ownership mismatch")
        identity = key_for(("artifact/generated-structure/v1", record.base_candidate,
                            record.proposed_candidate, artifact.occurrences, artifact.sample,
                            artifact.content, artifact.inputs))
        if (identity != artifact.identity or manifest(artifact.inputs.dependencies) != artifact.inputs
                or artifact.derived_from != (record.base_candidate,)):
            raise ValueError("Generated structure artifact identity/ownership mismatch")
        if any(occurrence.assembly != record.assembly for occurrence in artifact.occurrences):
            raise ValueError("Generated structure assembly ownership mismatch")
        _validate_sequences(record.chains, record.sequences, artifact.occurrences,
                            proposal.candidate, data)
        structure = data.get(record.generated_cif, _BYTES)
        if hashlib.sha256(structure).hexdigest() != record.structure_digest:
            raise ValueError("Retained generated structure digest mismatch")
        data.get(record.annotated_input, _BYTES)
        json.loads(data.get(record.metadata, _BYTES))
        records.append(record)
    return tuple(records)


def _publish(directory, stem, *, structure, annotated, metadata, details):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {f"{stem}.generated.cif": structure,
             f"{stem}.annotated.cif": annotated,
             f"{stem}.backend.json": metadata}
    index = dict(details)
    index["files"] = {name: {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
                      for name, value in files.items()}
    index["validation"] = "generated_unrefolded"
    files[f"{stem}.rfd4.generated.json"] = json.dumps(index, sort_keys=True, indent=2).encode() + b"\n"
    paths = []
    for name, value in files.items():
        path = directory / name
        if path.exists():
            if path.read_bytes() != value:
                raise ValueError(f"Refusing to overwrite a different generated artifact: {path}")
        else:
            with path.open("xb") as stream:
                stream.write(value)
        paths.append(path)
    return tuple(paths)


def materialize_generated_structure(proposal, data, destination) -> tuple[Path, ...]:
    paths = []
    for index, record in enumerate(read_generated_structures(proposal, data)):
        details = asdict(record)
        paths.extend(_publish(
            destination, f"{proposal.identity.digest}.{index}",
            structure=data.get(record.generated_cif, _BYTES),
            annotated=data.get(record.annotated_input, _BYTES),
            metadata=data.get(record.metadata, _BYTES), details=details,
        ))
    return tuple(paths)


def publish_generated_structure(result, destination, *, attempt="proposal") -> Path | None:
    """Publish a CLI attempt independently of disposable model-cache paths."""
    if not isinstance(result, GenerationResult):
        return None
    structure, annotated, metadata = _read_files(result)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    paths = _publish(directory, "rfd4", structure=structure, annotated=annotated,
                     metadata=metadata, details={
                         "attempt": str(attempt), "assembly": result.assembly,
                         "chains": [asdict(chain) for chain in result.chains],
                         "sequences": result.generated.sequences,
                         "request_digest": result.generated.request_digest,
                         "structure_digest": result.generated.structure_digest,
                     })
    return paths[-1]
