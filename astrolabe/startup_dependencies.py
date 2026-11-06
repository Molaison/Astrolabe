"""CPU-only startup dependency checks for a resolved, plain-dict run config.

Call ``check_startup_dependencies(cfg)`` before model initialization. Its return
value is the checked absolute DAlphaBall path (or None when not required); reuse
that path for PyRosetta's ``-holes:dalphaball`` option. The standalone resolver
uses exactly the same override/bundled policy, but does not launch the program.
No PyRosetta, model, configuration framework, or GPU imports are needed here.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import stat
import shlex
import subprocess
from collections.abc import Sequence


# Verified against losses/interaction.py, interface_holes.py and
# design_screening.py. The listed BindCraft fields all call the shared
# _bindcraft_interface_scores(), which computes dalphaball_sasa=1 BUNS even
# when the requested field is not itself the BUNS field. Other PyRosetta losses
# (including physical interface energy/SC and BindCraft residue counts) do not
# thereby require DAlphaBall.
DALPHABALL_LOSSES = frozenset({
    "chain_holes_diff",
    "interface_holes",
    "interface_buried_unsatisfied_hbonds",
    "ligand_local_holes",
    "ligand_buried_unsat_hbonds",
    "bindcraft_interface_sc",
    "bindcraft_interface_dG",
    "bindcraft_interface_dSASA",
    "bindcraft_interface_interface_hbonds",
    "bindcraft_interface_delta_unsat_hbonds",
    "bindcraft_binder_score",
    "bindcraft_surface_hydrophobicity",
})
_EXPRESSION_FIELDS = (
    "weighted_score_expression", "acceptance_threshold", "acceptance_gate",
    "early_stopping", "prefix_early_stopping", "_entrance",
)

# Protocol provenance: the local astrolabe/assets/DAlphaBall.gcc ELF's main
# (0x401150) reads NPOINTS, an integer, COORDS, that many x/y/z/r rows, then END.
# Its surf_vol branch (0x401424) calls get_surf_vol (0x4074f0), which prints
# "%d %.20lf %.20lf\n" (1-based index, surface, volume). adjust_nsphere__
# (0x408420) explicitly supports fewer than four spheres by adding internal
# dummy points; readjust_nsphere__ removes them before reporting. The actual
# debug_holes_DalphaBall_command.daball in the source checkout corroborates
# the input framing. This is a complete one-sphere request, NOT --help/EOF or
# an invented version flag. Addresses describe the inspected bundled build.
_PROBE_INPUT = "NPOINTS\n1\nCOORDS\n0 0 0 1\nEND\n"


class StartupDependencyError(ValueError):
    """A selected loss cannot use its required executable at startup."""


def _failure(path: Path, required_by: Sequence[str], reason: str) -> StartupDependencyError:
    return StartupDependencyError(
        f"DAlphaBall required by {', '.join(required_by)}; binary={path}: {reason}. "
        "Fix: set GH_DALPHABALL to a compatible executable or repair the bundled "
        "asset outside Astrolabe (including file/execute permissions and matching "
        "libgfortran/libquadmath runtime libraries). An invalid explicit "
        "GH_DALPHABALL is never replaced by the bundled binary."
    )


def resolve_dalphaball_path(*, required_by: Sequence[str] = ("GH_DALPHABALL",)) -> Path:
    """Resolve an explicit override, otherwise the bundled asset; never fallback.

    A present-but-empty override is an error too. Expansion reads but does not
    modify the current environment. Relative paths are anchored to the current
    directory, not searched on PATH. File/launch checks belong to preflight.
    """
    if "GH_DALPHABALL" in os.environ:
        raw = os.environ["GH_DALPHABALL"].strip()
        path = Path(os.path.abspath(os.path.expanduser(os.path.expandvars(raw))))
        if not raw:
            raise _failure(path, required_by, "GH_DALPHABALL is explicitly empty")
        return path
    return Path(__file__).resolve().parent / "assets" / "DAlphaBall.gcc"


def dalphaball_init_options(init_opts: str) -> str:
    """Add an available executable without fallback or permission changes.

    Required-loss preflight owns execution checks. Other PyRosetta metrics may
    run without DAlphaBall, so an unavailable optional binary is omitted.
    Explicit PyRosetta options retain precedence.
    """
    tokens = shlex.split(init_opts)
    flag = "-holes:dalphaball"
    if flag in tokens:
        index = tokens.index(flag)
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            return init_opts
        raise StartupDependencyError("-holes:dalphaball requires an executable path")
    if "GH_DALPHABALL" in os.environ and not os.environ["GH_DALPHABALL"].strip():
        return init_opts
    path = resolve_dalphaball_path()
    if path.is_file() and os.access(path, os.X_OK):
        return f"{init_opts} {flag} {shlex.quote(str(path))}"
    return init_opts


def _expression_names(expression: str) -> set[str]:
    tree = ast.parse(expression, mode="eval")
    return {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }


def dalphaball_required_by(cfg: dict) -> tuple[str, ...]:
    """Identify selected DAlphaBall losses without importing the loss registry.

    Match LossEngine semantics: non-null explicit loss entries are computed even
    when absent from the score expression, unless a stage has active_loss_terms.
    Check every non-null configured stage, since later stages must also work.
    A stage's active_loss_terms does not inherit from default (StageManager).
    Expression fields do inherit when absent; reachable expression_terms and
    implicit registered loss names in expressions are included. Configuration
    shape/syntax validation remains the caller's responsibility.
    """
    loss = cfg.get("loss") or {}
    terms = loss.get("loss_terms") or {}
    expressions = loss.get("expression_terms") or {}
    stages = cfg.get("stages") or {}
    default = stages.get("default") or {}
    contexts = [(f"stages.{name}", stage) for name, stage in stages.items()
                if stage is not None] or [("", {})]
    required = []
    for stage_path, stage in contexts:
        active = stage.get("active_loss_terms")
        for alias, spec in terms.items():
            if (spec is not None and spec.get("_type") in DALPHABALL_LOSSES
                    and (active is None or alias in active)):
                required.append(f"loss.loss_terms.{alias}._type ({spec['_type']})")

        pending = []
        for field in _EXPRESSION_FIELDS:
            owner = stage if field in stage else default
            expression = owner.get(field)
            if isinstance(expression, str) and expression:
                owner_path = stage_path if field in stage else "stages.default"
                pending.append((f"{owner_path}.{field}", expression))
        visited = set()
        while pending:
            field_path, expression = pending.pop()
            for name in sorted(_expression_names(expression)):
                if name in expressions and name not in visited:
                    visited.add(name)
                    pending.append((f"loss.expression_terms.{name}", expressions[name]))
                elif (name not in terms and name not in expressions
                      and name in DALPHABALL_LOSSES
                      and (active is None or name in active)):
                    required.append(f"{field_path} ({name})")
    return tuple(dict.fromkeys(required))


def _diagnostic(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "").strip()[:2000]


def check_startup_dependencies(cfg: dict, *, timeout: float = 5.0) -> Path | None:
    """Fail fast only for selected DAlphaBall losses; return the checked path.

    The probe inherits the process environment unchanged, uses no shell and has
    a finite positive timeout in seconds. subprocess.run kills and reaps the
    direct executable on timeout. Success requires a zero exit and a valid
    positive, finite one-sphere surface/volume response, not merely existence
    or an executable mode bit. Nothing is chmod'ed or installed here.
    """
    required = dalphaball_required_by(cfg)
    if not required:
        return None
    path = resolve_dalphaball_path(required_by=required)
    if not math.isfinite(timeout) or timeout <= 0:
        raise _failure(path, required, "probe timeout must be finite and positive")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise _failure(path, required, f"cannot stat executable: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise _failure(path, required, "path is not a regular file")
    if not os.access(path, os.X_OK):
        raise _failure(path, required, "file is not executable (execute permission denied)")
    try:
        result = subprocess.run(
            [str(path), "surf_vol"], input=_PROBE_INPUT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _diagnostic(exc.stderr) or _diagnostic(exc.stdout)
        raise _failure(path, required, f"probe timed out after {timeout:g}s; {detail}") from exc
    except OSError as exc:
        raise _failure(
            path, required,
            f"cannot execute/load file: {exc}; check ELF architecture/interpreter "
            "and shared-library availability even when the file exists",
        ) from exc
    if result.returncode != 0:
        detail = _diagnostic(result.stderr) or _diagnostic(result.stdout) or "no diagnostic output"
        raise _failure(path, required, f"surf_vol probe exited {result.returncode}: {detail}")
    fields = result.stdout.split()
    try:
        valid = (
            len(fields) == 3 and int(fields[0]) == 1
            and all(math.isfinite(float(value)) and float(value) > 0
                    for value in fields[1:])
        )
    except ValueError:
        valid = False
    if not valid:
        raise _failure(
            path, required,
            "invalid surf_vol one-sphere response (expected index 1, positive finite "
            f"surface and volume); stdout={_diagnostic(result.stdout)!r}; "
            f"stderr={_diagnostic(result.stderr)!r}",
        )
    return path
