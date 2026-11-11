"""Locate AF3 confidence without guessing an evaluator or prediction identity."""

from pathlib import Path


def resolve_af3_confidence_path(
    structure_path=None, *, raw_model_results_dir=None, identifier=None, oligo_name=None
) -> Path | None:
    """Prefer the structure's native output; discover only within its raw root.

    ``identifier`` is a legacy fallback for callers without a native model name.
    An alias-qualified structure never falls back to a different alias or root.
    Missing files return None; conflicting identities or candidates raise.
    """
    raw_root = Path(raw_model_results_dir) if raw_model_results_dir is not None else None
    if structure_path is not None:
        path = Path(structure_path)
        suffix = "_model.cif"
        alias_structure = (
            path.parent.parent.name == "evaluators"
            and path.parent.parent.parent.name == "structures"
        )
        if not path.name.endswith(suffix) and (
            alias_structure or "raw_model_results" in path.parts
        ):
            raise ValueError(f"Not an AF3 native model filename: {path}")
        if path.name.endswith(suffix):
            identifier = path.name.removesuffix(suffix)
            if oligo_name is not None and not (
                identifier == oligo_name or identifier.endswith(f"_{oligo_name}")
            ):
                raise ValueError(f"AF3 oligomer identifier mismatch for {oligo_name}: {path}")
            filename = f"{identifier}_confidences.json"
            parent = path.parent
            # Native AF3 output, including saved baseline outputs.
            if "raw_model_results" in path.parts:
                if parent.parent.name != "alphafold3":
                    raise ValueError(f"Not an AF3 native structure: {path}")
                if parent.name != identifier:
                    raise ValueError(f"AF3 prediction identifier mismatch: {path}")
            if parent.parent.name == "alphafold3":
                if parent.name != identifier:
                    raise ValueError(f"AF3 prediction identifier mismatch: {path}")
                candidate = parent / filename
                return candidate if candidate.is_file() else None
            if alias_structure:
                raw_root = parent.parent.parent.parent / "raw_model_results"
                candidate = raw_root / "evaluators" / parent.name / "alphafold3" / identifier / filename
                return candidate if candidate.is_file() else None
            if parent.name == "structures":
                raw_root = parent.parent / "raw_model_results"
            else:
                candidate = parent / filename
                return candidate if candidate.is_file() else None

    if raw_root is None or identifier is None:
        return None
    if not identifier or Path(identifier).name != identifier or identifier in (".", ".."):
        raise ValueError(f"Invalid AF3 prediction identifier: {identifier!r}")
    filename = f"{identifier}_confidences.json"
    candidates = [raw_root / "alphafold3" / identifier / filename]
    candidates.extend((raw_root / "evaluators").glob(f"*/alphafold3/{identifier}/{filename}"))
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous AF3 confidence for {identifier}: {candidates}")
    return candidates[0] if candidates else None
