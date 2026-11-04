# Preserved historical GPU-control identity review

**DONE for the historical CPU checker/source review; no GPU certification.**
This package closes an evidence-index gap. It preserves all 16 original files
from Agent C's `control/f102-gpu-science-review/`, without editing their contents,
rerunning tests, generating model output, or changing production source.

The actual review compared the earlier harness
`1b7a1bdc32dda273fac6b4e7bebad8a5a9c1f0ca` with the repair
`4889f7b7c08828a99b729719f0c668eaaf4d2396`. These are historical source anchors;
the directory's final-HEAD prefix does not claim that those experiments ran on
1bdaf3a or that a real GPU ran. Current final GPU admission evidence belongs to
Agent B's separate supplement and Agent A's accounting proof.

## Reproduced defect and repair

On the first source, a normal synthetic fixture passed. Changing the child
seed, mode, source HEAD, request indices or request phases could still produce
PASS with certified effective-throughput fields. The controller's own rows
were checked, but the child identity/phase fields were not fully checked.
The combined original counterexample and the five separate false positives
are preserved. On 4889f7b the same five corruptions return INVALID and omit
effective throughput; the unchanged positive control still passes.

Direct reviewable records:

- [Combined original false positive](child-identity-before.json).
- [Five separate original false positives](child-identity-before-separate.json).
- [Five rejected corruptions and positive control after repair](child-identity-after-4889f7b.json).
- [Unchanged reproducer](probe_child_identity.py).
- [Independent fixed-source review](review-b-4889f7b.md).
- [Original protocol review](protocol-review.md).

The archive also retains the original 52/62-case pytest logs and JUnit, literal
protocol audit script/results, manifests and the separate recovered-Runtime
owner probe. These distinct review scopes are not summed as unique test
coverage. Earlier protocol snapshots remain historical snapshots; they are not
silently relabelled as the eventual frozen protocol.

This is a checker-contract defect with synthetic CPU evidence, not a measured
GPU scientific mismatch or actual throughput result. The protocol review
preserves complete-input/seed/decode/native-owner constraints. Neither it nor
the repaired checker establishes pretrained inference, GPU output/RNG parity,
full registry acceptance/stage behavior, or revision-attributed acceleration.

## Complete original archive

`gpu-identity-review-raw.tar.xz` contains **16 files /65,791 original bytes** under
`f102-gpu-science-review/`. It is **16,268 bytes**, SHA-256:

```text
c480ca6a60b67172a52fc8dd2905c6e577efaae241d74474c8f99fab47409883
```

`RAW_MANIFEST.json` gives every original path, byte length and SHA-256. Each
archived member was reread and checked against that manifest; the source files
were rehashed and remain unchanged. The direct files above are byte-identical
convenience copies. Overall historical CPU wall/CPU budget was not measured;
it remains unknown. No new experiment was run while packaging.

Verify and extract into a new directory:

```bash
printf '%s  %s\n' c480ca6a60b67172a52fc8dd2905c6e577efaae241d74474c8f99fab47409883 gpu-identity-review-raw.tar.xz | sha256sum -c -
mkdir extracted
tar -xJf gpu-identity-review-raw.tar.xz -C extracted
python - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('extracted/f102-gpu-science-review')
manifest = json.loads(Path('RAW_MANIFEST.json').read_text())
assert set(manifest) == {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
for name, identity in manifest.items():
    data = (root / name).read_bytes()
    assert len(data) == identity['bytes']
    assert hashlib.sha256(data).hexdigest() == identity['sha256'], name
print('PASS: all 16 original files preserved')
PY
```

Scientific or checker reruns, if separately requested, require the exact source
anchors and original dependencies recorded in `review-b-4889f7b.md`. Reading
or verifying this archive runs neither a model nor a benchmark.
