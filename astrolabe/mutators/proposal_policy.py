"""Optional common probability-space constraints for mutation proposals."""
from dataclasses import dataclass
import math
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True)
class ProposalPolicy:
    allowed_residues: tuple[str, ...] = ()
    disallowed_residues: tuple[str, ...] = ()
    bias: tuple[tuple[str, float], ...] = ()
    temperature: float = 1.0
    max_mutations: int | None = None

    @classmethod
    def from_config(cls, config):
        if config is None:
            return None
        config = dict(config)
        unknown = set(config) - {'allowed_residues','disallowed_residues','bias','temperature','max_mutations'}
        if unknown:
            raise ValueError(f'unknown proposal_policy settings: {sorted(unknown)}')
        bias = config.get('bias') or {}
        if not isinstance(bias, Mapping):
            raise ValueError('proposal_policy.bias must map residues to log biases')
        return cls(tuple(str(x).upper() for x in (config.get('allowed_residues') or ())),
                   tuple(str(x).upper() for x in (config.get('disallowed_residues') or ())),
                   tuple((str(k).upper(), float(v)) for k, v in bias.items()),
                   float(config.get('temperature', 1.0)),
                   None if config.get('max_mutations') is None else int(config['max_mutations']))

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('proposal_policy.temperature must be finite and positive')
        if self.max_mutations is not None and self.max_mutations < 0:
            raise ValueError('proposal_policy.max_mutations must be nonnegative')
        if any(not math.isfinite(value) for _, value in self.bias):
            raise ValueError('proposal_policy.bias values must be finite')
        if len({key for key, _ in self.bias}) != len(self.bias):
            raise ValueError('proposal_policy.bias residues must be unique')

    @property
    def changes_probabilities(self):
        return bool(self.allowed_residues or self.disallowed_residues or self.bias or self.temperature != 1.0)

    def apply(self, probabilities, alphabet, excluded_positions=()):
        if not self.changes_probabilities:
            return probabilities
        alphabet = tuple(alphabet)
        if len(set(alphabet)) != len(alphabet):
            raise ValueError('proposal alphabet must have unique residues')
        configured = set(self.allowed_residues) | set(self.disallowed_residues) | {key for key, _ in self.bias}
        if not configured <= set(alphabet):
            raise ValueError('proposal_policy contains residues outside the backend alphabet')
        result = np.array(probabilities, dtype=float, copy=True)
        if result.ndim < 2 or result.shape[-1] != len(alphabet) or not np.isfinite(result).all() or (result < 0).any():
            raise ValueError('invalid proposal probabilities or alphabet')
        excluded = set(int(pos) for pos in excluded_positions)
        if any(pos < 0 or pos >= result.shape[-2] for pos in excluded):
            raise ValueError('proposal mask position is out of range')
        log_bias = dict(self.bias)
        allowed = set(self.allowed_residues) if self.allowed_residues else set(alphabet)
        allowed -= set(self.disallowed_residues)
        scores = [log_bias.get(aa, 0.0) / self.temperature for aa in alphabet if aa in allowed]
        if not scores or not all(math.isfinite(score) for score in scores):
            raise ValueError('proposal_policy has no finite allowed residues')
        maximum = max(scores)
        weights = np.array([math.exp(log_bias.get(aa, 0.0) / self.temperature - maximum) if aa in allowed else 0.0 for aa in alphabet])
        for pos in range(result.shape[-2]):
            if pos in excluded:
                continue
            row = result[..., pos, :]
            original_total = row.sum(axis=-1, keepdims=True)
            with np.errstate(over="ignore", invalid="ignore"):
                transformed = np.power(row, 1.0 / self.temperature) * weights
            total = transformed.sum(axis=-1, keepdims=True)
            if not np.isfinite(transformed).all() or not np.isfinite(total).all():
                raise ValueError(f'proposal_policy produced nonfinite values at position {pos}')
            if np.any((original_total > 0) & (total <= 0)):
                raise ValueError(f'proposal_policy removes all probability at position {pos}')
            result[..., pos, :] = np.divide(transformed, total, out=np.zeros_like(transformed), where=total > 0)
        return result


def mutation_budget(base, weight, *, existing_cap=None, policy=None):
    """The historical round operation, with optional explicit caps."""
    budget = round(base * weight)
    if existing_cap is not None:
        budget = min(budget, existing_cap)
    if policy is not None and policy.max_mutations is not None:
        budget = min(budget, policy.max_mutations)
    return budget
