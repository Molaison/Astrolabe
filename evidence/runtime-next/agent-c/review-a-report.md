# Independent interrogate: Agent A deadline and delegation ownership

**Reviewer:** Agent C, independent from author A. **Final verdict:** the shared deadline, delegation ownership and entry-lock budget corrections pass the independently reproduced scoped gates through `eac3081`. Both additional preexisting defects found during review were corrected in separate author commits and independently retested. Bookkeeping and cleanup may still delay the return after timeout; the verified guarantee is no new dispatch after the applicable admission budget expires.

## Intent and exact artifacts

The user requires scientific requests not to be dispatched after their applicable deadline, no replay, and resource/ownership integrity under timeout. Review scope was explicitly assigned by the coordinator:

- `54031feb7bbfb018b2e64a0fb5b444b95ad8f3c1`: one admission deadline across worker call lock, pending resource admission, compute admission and final worker dispatch.
- `00a8b688db3a1e876d014212a614273c14ef29b2`: preserve an active caller's delegation when another caller times out waiting for the same worker lock.
- `eac3081afc6eca374ffc8ae37da8a37a6afd43ec`: start the same admission budget at `State.call` entry, preserving request identity/lock ordering.
- The intervening `1055b86` modifies only the validation subprocess environment. Its broker blob is identical to `54031fe`; it was not reviewed as a model/runtime change.

No files in Agent A's worktree were changed by this reviewer. Real broker `State`, resource ledger and Python locks were exercised directly, with explicitly controlled worker boundaries. There was no real scientific worker/model, UNIX IPC or GPU execution. Unsupported OS process observations were not claimed as tested.

## 1. Shared admission deadline: independently reproduced correction

Author A's probe was independently executed by C against the original broker source and the revised broker source. The original source was read from Agent C's checkout at `8c987ed`; its broker blob `7d71de224c5d7659dff560440645ce092fa9efd6` was checked equal to d66. C's unrelated producer changes do not touch that module.

| Measured control-plane behavior | Original d66 broker source | `54031fe` |
|---|---:|---:|
| Request timeout | 1.0 s | 1.0 s |
| Pending queue wait | 0.6003 s | 0.6003 s |
| Total time until outcome | 1.3012 s | 1.0005 s |
| Worker boundary reached | Yes, around 1.3010 s | No |
| Worker received fresh minimum timeout | 0.001 s | No dispatch |
| Reservation/busy/waiting retained after outcome | No | No |
| Same request ID rejected on reuse | Yes | Yes |

This validates the intended queue-budget correction; these times are timeout observations, not a throughput benchmark. Raw evidence is `review-a-deadline-original/result.json` and `review-a-deadline-fixed/result.json`, with corresponding runtime JSONL. The fixed evidence records HEAD `54031fe`, clean broker source, SHA256 `9247a3a01d06cc2e37a6e4ab2bebfcf14eee59ff163918481892eeaafdbc48c7`.

## 2. Verified ownership finding, separately fixed and independently retested

**Finding:** A waiting caller that never acquired `record['call_lock']` still executed unconditional `record['delegated'] = None` in `finally`. An active first request retained its compute reservation but lost its delegated metadata. When its controlled worker returned successfully, `record['delegated']['nested']` failed with `TypeError: 'NoneType' object is not subscriptable`.

C wrote `probe_delegate_timeout.py` independently of A's tests. It runs a no-waiter positive control and a real concurrent caller/waiter control. It uses real State admission/cleanup; only worker execution and post-response process observations are controlled. Both d66 broker source and `54031fe` broker source reproduced the bug, so **this is preexisting, not introduced by the deadline patch**. Root also independently reproduced it, then authorized A's separate fix.

| Observation | d66 / `54031fe` broker source | `00a8b68` |
|---|---:|---:|
| No-waiter request returns successfully | Yes | Yes |
| Second caller times out on worker lock | Yes | Yes |
| First delegation survives second caller's timeout | No | Yes |
| First request returns successfully after worker reply | No: TypeError | Yes |
| Final busy / pending count | 0 / 0 | 0 / 0 |

The fix changes only cleanup ownership: clear delegation only when `acquired` is true. It does not release the active caller's compute reservation or grant permission to replay the waiting request. C reran the same independent probe on exact commit `00a8b68`; `review-a-delegate-00a8b68.json` contains the corrected positive and concurrent controls. Original and pre-fix-current raw JSON remain preserved.

## 3. Verified entry-lock budget defect, separately fixed and retested

A separate independent probe, `probe_entry_lock_deadline.py`, blocks `State.lock` while `_track_request` is about to acquire it. `State.call` assigns `start` and `deadline` only after `_track_request` returns. Therefore a **30 ms timeout still reaches dispatch around 120 ms after entry**, with about 29.9 ms freshly available at the worker boundary.

This was reproduced against d66 source and then exact `00a8b68`. Raw output: `review-a-entry-lock-original.json` and `review-a-entry-lock-00a8b68.json`. The latter records elapsed 0.1205 s, dispatch at 0.1205 s, worker timeout 0.0299 s, and complete reservation cleanup. This is another preexisting limitation, not a regression caused by the reviewed commits.

Root independently verified this finding and authorized A's separate `eac3081` correction, moving `start` to function entry without changing request identity checks or lock/cleanup ordering. C reran the original negative case on that exact commit: the 120 ms bookkeeping wait now ends in TimeoutError with **no dispatch**. An added positive control uses a 500 ms budget and the same 120 ms lock hold: dispatch remains allowed and receives only 379 ms, proving elapsed entry time is charged without spuriously rejecting live work. Raw outputs are `review-a-entry-lock-eac3081.json` and `review-a-entry-lock-live-eac3081.json`.

The delegation probe was also rerun on `eac3081`; the first caller succeeds with or without a timed-out waiter, and final busy/pending counts are zero (`review-a-delegate-eac3081.json`). The admitted deadline includes entry bookkeeping time, but lock waits and cleanup may delay the wall-clock return; the patch deliberately does not promise a hard real-time cancellation bound. No source change was made by C.

## Regression results and dispositions

C independently ran A's deadline tests at `54031fe`: 6 passed. After the ownership fix, C ran `tests/test_runtime_broker_deadline.py` and `tests/test_runtime_contracts.py` on exact `00a8b68`: **16 passed in 1.23 s**, saved in `review-a-00a8b68-tests.log`.

Final combined gate on `eac3081`: **18 passed in 1.48 s**, saved in `review-a-eac3081-tests.log`.

Retained findings: one ownership defect and one entry-lock timeout budget defect, both now corrected in separate commits and independently verified. Discarded claim: the ownership defect was introduced by the new deadline correction (refuted by baseline). No other new defect in the two reviewed changes was established by the executed probes. A green suite is not certification of unexecuted IPC/GPU failure modes.

A provisional entry-lock observation made while A was preparing its follow-up had a working-tree source hash not matching its then-current HEAD; it is retained as raw history only. Conclusions use the later exact `00a8b68` observation instead.

## Reproduction

From Agent C's checkout, with the reviewed Agent A checkout pinned to the recorded commit:

```sh
../venv/bin/python ../agent-a/evidence/runtime-next/agent-a/probe_deadline.py --repo ../agent-a --out evidence/runtime-next/agent-c/NEW-deadline-review
../venv/bin/python evidence/runtime-next/agent-c/probe_delegate_timeout.py --repo ../agent-a --out evidence/runtime-next/agent-c/NEW-delegate-review.json
../venv/bin/python evidence/runtime-next/agent-c/probe_entry_lock_deadline.py --repo ../agent-a --out evidence/runtime-next/agent-c/NEW-entry-review.json
```

The author probe requires a new output directory and never overwrites an existing evidence directory. Root has independent copies/results and owns the integration decision. Review used a real separate agent, not a claim of different model families.
