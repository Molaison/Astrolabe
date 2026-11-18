"""Lightweight loss-configuration fields shared by preflight and consumers."""

from collections.abc import Sequence


INTRINSIC_PARAMS = frozenset({"oligomer_specific"})


def loss_oligomers(options):
    field = "_for" if "_for" in options else "oligomer_specific"
    if "_for" in options and "oligomer_specific" in options:
        raise ValueError("use only one of _for and oligomer_specific")
    value = options.get(field)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, Sequence) or not value or not all(
        isinstance(name, str) and name.strip() for name in value
    ):
        raise ValueError("expected an oligomer name, comma-separated names, or a nonempty list of names")
    return [name.strip() for name in value]


def loss_parameter_errors(options, *, path, kind, required, optional):
    required = set(required or ())
    optional = set(optional or ())
    for key in sorted(required):
        if options.get(key) is None:
            yield f"{path}.{key}: required by loss {kind!r}"
    unexpected = set(options) - {"_type", "_for"} - INTRINSIC_PARAMS - required - optional
    for key in sorted(unexpected):
        yield f"{path}.{key}: invalid parameter for loss {kind!r}"
