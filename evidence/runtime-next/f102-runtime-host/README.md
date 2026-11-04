# Real f102 CPU process contracts

Source `60684bd4fefa6e9b7282176e8b5af7700ef98093`, user-requested server directory,
real SIF Python 3.11.14/Torch 2.7.1+cu126 with all accelerators explicitly hidden.
**309 passed, zero failures/skips**, 148.045807427 seconds in the actual Linux
process/AF_UNIX environment. The original process identity, observer, CPU failure
fencing, saturation, admission cancellation and retained ownership gates ran;
`ASTROLABE_EXPECT_GUARDRAILS=1` remained enabled. This is CPU evidence, not CUDA inference.

All observed process identities released. Execution had a 300-second wall budget
and 60-MB output budget. The full archive is 115,008 bytes with SHA256
`8f5e81c1860d5a7d823a095f8ca6029709e0e9a4a50ce10461d3722ed005f1e3`.
It preserves 265 regular files/2,155,405 bytes and a manifest. One fixture symlink
is represented in manifest metadata, not dereferenced; its target file is included.
Root downloaded and checked all file digests independently.

The source clone initially inherited a Git object-alternate warning. After this
run, only the new task clone was repacked, the alternate saved/detached, and Git
connectivity plus all required source commits independently verified. The raw
warning is retained; it is not presented as a test failure or silently removed.
Original neighbouring worktrees and Git objects were never modified.
