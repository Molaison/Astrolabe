"""Generic repeat planning and materialization.

The application owns one execution loop.  This module only describes each
repeat and constructs its configuration from user-facing repeat controls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf, open_dict


_MAX_U32 = 2**32 - 1


def _container(cfg: Any) -> dict[str, Any]:
    if cfg is None or not hasattr(cfg, "get"):
        raise TypeError("configuration must be a mapping or composed DictConfig")
    conf = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    value = OmegaConf.to_container(conf, resolve=False)
    if not isinstance(value, dict):
        raise TypeError("configuration must resolve to a mapping")
    for key in ("repeats", "seed", "out_dir", "task_name"):
        node = OmegaConf.select(
            conf,
            key,
            default=None,
            throw_on_missing=False,
        )
        if node is not None:
            value[key] = node
    for key in ("search", "repeat_plan"):
        node = OmegaConf.select(
            conf,
            key,
            default=None,
            throw_on_missing=False,
        )
        if node is not None:
            value[key] = OmegaConf.to_container(
                node,
                resolve=True,
                throw_on_missing=True,
            )
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _integer(value: Any, *, name: str) -> int:
    decimal_string = False
    if type(value) is str:
        digits = value[1:] if value[:1] in "+-" else value
        decimal_string = bool(digits) and digits.isascii() and digits.isdecimal()
    if type(value) is not int and not decimal_string:
        raise ValueError(f"Invalid {name}={value!r}; expected a decimal integer.")
    return int(value)


def _seed(value: Any, *, name: str) -> int:
    seed = _integer(value, name=name)
    if not 1 <= seed <= _MAX_U32:
        raise ValueError(
            f"Invalid seed sequence: {name} must be in 1..{_MAX_U32}, got {seed}"
        )
    return seed


def _optional_seed(value: Any, *, name: str) -> int | None:
    if value in (None, "", "null", "None"):
        return None
    return _seed(value, name=name)


def _as_start(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"sequence": value}
    if not isinstance(value, Mapping):
        raise TypeError("repeat start entries must be sequence strings or mappings")
    result = dict(value)
    sequence = result.get("sequence", result.get("seq"))
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("repeat start mapping requires a non-empty sequence")
    result["sequence"] = sequence
    return result


def _python_random_sequence(
    specification: Mapping[str, Any],
    *,
    seed: int,
) -> str:
    alphabet = specification.get("alphabet")
    if not isinstance(alphabet, str) or not alphabet:
        raise ValueError("repeat initial_sequence.alphabet must be a non-empty string")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("repeat initial_sequence.alphabet must not contain duplicates")
    length = specification.get("length")
    if not isinstance(length, Sequence) or isinstance(length, (str, bytes)):
        raise ValueError("repeat initial_sequence.length must be [minimum, maximum]")
    if len(length) != 2:
        raise ValueError("repeat initial_sequence.length must be [minimum, maximum]")
    minimum = _integer(length[0], name="initial_sequence.length minimum")
    maximum = _integer(length[1], name="initial_sequence.length maximum")
    if minimum < 1 or maximum < minimum:
        raise ValueError(
            "repeat initial_sequence.length requires 1 <= minimum <= maximum"
        )
    rng = random.Random(seed)
    return "".join(rng.choices(alphabet, k=rng.randint(minimum, maximum)))


INITIAL_SEQUENCE_GENERATORS: dict[str, Callable[[Mapping[str, Any], int], str]] = {
    "python_random": lambda specification, seed: _python_random_sequence(
        specification,
        seed=seed,
    ),
}


@dataclass(frozen=True)
class RepeatRun:
    """One planned execution in the application repeat loop."""

    index: int
    total: int
    seed: int
    output: Path | None = None
    start: Mapping[str, Any] | None = None
    initial_sequence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("RepeatRun.index must be a nonnegative integer")
        if type(self.total) is not int or self.total <= self.index:
            raise ValueError("RepeatRun.total must include the repeat index")
        _seed(self.seed, name="RepeatRun.seed")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        if self.start is not None:
            object.__setattr__(self, "start", dict(self.start))
        if self.initial_sequence is not None:
            object.__setattr__(
                self,
                "initial_sequence",
                dict(self.initial_sequence),
            )

    @property
    def number(self) -> int:
        return self.index + 1

    @property
    def repeat_index(self) -> int:
        return self.index

    @property
    def count(self) -> int:
        return self.total

    @property
    def out_dir(self) -> Path | None:
        return self.output


def _selected_starts(
    *,
    search: Mapping[str, Any],
    repeat_plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    raw_starts = repeat_plan.get("starts")
    if raw_starts is None and str(search.get("initialization", "")).lower() == "archived":
        raw_starts = search.get("archived_starts")
        if raw_starts is None and search.get("start") is not None:
            raw_starts = [search["start"]]
    if isinstance(raw_starts, (str, Mapping)):
        raw_starts = [raw_starts]
    starts = [] if raw_starts is None else list(raw_starts)
    cursor = _integer(
        repeat_plan.get("start_cursor", search.get("archived_start", 0)),
        name="repeat_plan.start_cursor",
    )
    if cursor < 0:
        raise ValueError("repeat_plan.start_cursor must be nonnegative")
    return [_as_start(item) for item in starts], cursor


def _repeat_count(cfg: Mapping[str, Any], repeat_plan: Mapping[str, Any]) -> int:
    raw = repeat_plan.get("count", cfg.get("repeats", 1))
    count = _integer(raw, name="repeats")
    if count <= 0:
        raise ValueError(f"Invalid repeats={raw!r}; expected a positive decimal integer.")
    configured = cfg.get("repeats", 1)
    if repeat_plan.get("count") is not None:
        cfg_count = _integer(configured, name="repeats")
        if cfg_count != count:
            raise ValueError(
                "repeat_plan.count and top-level repeats disagree; use one scheduler"
            )
    return count


def plan_repeat_runs(base_cfg: Any) -> tuple[RepeatRun, ...]:
    """Validate and plan every repeat before execution starts."""

    cfg = _container(base_cfg)
    repeat_plan = _mapping(cfg.get("repeat_plan"), name="repeat_plan")
    search = _mapping(cfg.get("search"), name="search")
    count = _repeat_count(cfg, repeat_plan)
    starts, cursor = _selected_starts(search=search, repeat_plan=repeat_plan)
    if starts and cursor + count > len(starts):
        raise ValueError(
            "--num-runs exceeds the declared repeat starts: "
            f"requested {count}, available {len(starts) - cursor}"
        )

    explicit_override = _optional_seed(
        repeat_plan.get("seed_override"),
        name="repeat_plan.seed_override",
    )
    start_seed = _optional_seed(
        repeat_plan.get("start_seed"),
        name="repeat_plan.start_seed",
    )
    base_seed = _optional_seed(cfg.get("seed"), name="seed")
    explicit_seeds = repeat_plan.get("seeds")
    if explicit_seeds is not None:
        explicit_seeds = list(explicit_seeds)
        if len(explicit_seeds) < count:
            raise ValueError("repeat_plan.seeds must provide one seed per repeat")

    out_dir_value = cfg.get("out_dir")
    out_dir = None
    if out_dir_value not in (None, "", "null", "None"):
        out_dir = Path(str(out_dir_value)).expanduser()
    output_pattern = repeat_plan.get("output", repeat_plan.get("output_pattern"))
    if output_pattern is not None and not isinstance(output_pattern, str):
        raise TypeError("repeat_plan.output must be a string path template")
    if output_pattern and out_dir is None:
        raise ValueError("repeat_plan.output requires an output directory")

    initial_spec = repeat_plan.get("initial_sequence")
    if initial_spec is not None and starts:
        raise ValueError("repeat_plan cannot combine initial_sequence with repeat starts")

    task_name = str(cfg.get("task_name") or "")
    planned: list[RepeatRun] = []
    for index in range(count):
        start = starts[cursor + index] if starts else None
        if explicit_override is not None:
            seed = explicit_override + index
        elif explicit_seeds is not None:
            seed = _seed(explicit_seeds[index], name=f"repeat_plan.seeds[{index}]")
        elif start is not None and start.get("seed") is not None:
            seed = _seed(start["seed"], name=f"repeat start {cursor + index} seed")
        elif start_seed is not None:
            seed = start_seed + index
        elif base_seed is not None:
            seed = base_seed + index
        else:
            digest = hashlib.sha256(
                f"{task_name}:{index}".encode("utf-8")
            ).hexdigest()
            seed = (int(digest[:8], 16) % 1_000_000) + 1
        _seed(seed, name=f"repeat {index + 1} seed")

        output = out_dir
        if output_pattern:
            relative = output_pattern.format(
                index=index,
                repeat=index + 1,
                seed=seed,
            )
            output = out_dir / relative

        initial_sequence = None
        if initial_spec is not None:
            specification = _mapping(
                initial_spec,
                name="repeat_plan.initial_sequence",
            )
            owner = str(specification.pop("owner", "python_random")).strip()
            generator = INITIAL_SEQUENCE_GENERATORS.get(owner)
            if generator is None:
                raise ValueError(f"Unknown initial sequence generator {owner!r}")
            target = str(specification.pop("target", "A")).strip()
            if not target:
                raise ValueError("repeat_plan.initial_sequence.target must be non-empty")
            initial_sequence = {
                "target": target,
                "value": generator(specification, seed),
            }

        planned.append(
            RepeatRun(
                index=index,
                total=count,
                seed=seed,
                output=output,
                start=start,
                initial_sequence=initial_sequence,
            )
        )
    return tuple(planned)


def materialize_repeat(base_cfg: Any, repeat: RepeatRun) -> Any:
    """Return one independent config for an already validated repeat."""

    if not isinstance(repeat, RepeatRun):
        raise TypeError("repeat must be a RepeatRun")
    base_obj = OmegaConf.to_container(base_cfg, resolve=False)
    run_cfg = OmegaConf.create(base_obj)
    with open_dict(run_cfg):
        run_cfg.repeats = repeat.total
        run_cfg.repeat_index = repeat.index
        run_cfg.seed = repeat.seed
        if repeat.output is not None:
            run_cfg.out_dir = str(repeat.output)
        if repeat.start is not None:
            search = run_cfg.get("search")
            if not hasattr(search, "get"):
                raise ValueError(
                    "repeat start requires an explicit search route; "
                    "materialize_repeat will not create search"
                )
            if not str(search.get("algorithm", "")).strip():
                raise ValueError(
                    "repeat start requires an explicit search.algorithm"
                )
            if not str(search.get("owner", "")).strip():
                raise ValueError(
                    "repeat start requires an explicit search.owner"
                )
            run_cfg.search.start = dict(repeat.start)
            run_cfg.search.archived_start = 0
        if repeat.initial_sequence is not None:
            target = str(repeat.initial_sequence["target"])
            value = str(repeat.initial_sequence["value"])
            if run_cfg.get("seqs") is None:
                run_cfg.seqs = {}
            run_cfg.seqs[target] = value

        raw_dir = run_cfg.get("raw_model_results_dir")
        if raw_dir not in (None, "", "null", "None"):
            run_cfg.raw_model_results_dir = (
                Path(str(raw_dir)).expanduser()
                / f"repeat_{repeat.number:03d}"
            )
    return run_cfg


__all__ = [
    "INITIAL_SEQUENCE_GENERATORS",
    "RepeatRun",
    "materialize_repeat",
    "plan_repeat_runs",
]
