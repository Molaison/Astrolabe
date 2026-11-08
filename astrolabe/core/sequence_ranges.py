"""Structure sequence selectors shared by startup and Protomer consumers."""
from collections import defaultdict
import re


def parse_ranges(s: str):
    if s is None:
        raise ValueError("Missing chain/range statement (expected like 'A' or 'A1-10').")
    result = defaultdict(list)
    s = str(s).strip()
    if not s:
        raise ValueError("Empty chain/range statement (expected like 'A' or 'A1-10').")
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        match_chain_only = re.fullmatch(r"([A-Za-z])", item)
        if match_chain_only:
            result[match_chain_only.group(1)] = []
            continue
        match = re.fullmatch(r"([A-Za-z])(?::?)(\d+)(?:-(\d+))?", item)
        if not match:
            raise ValueError(f"Invalid format: {item}")
        prefix = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        if end < start:
            raise ValueError(f"Invalid residue range {item!r}: end < start")
        result[prefix].extend(range(start, end + 1))
    if not result:
        raise ValueError("No chain/range selected")
    return dict(result)
