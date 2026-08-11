# Scratch abort on the non-LTS stack: A750 reproduction attempt (results)

**Date:** 2026-08-10, local machine `intel-G501`.
**Stack:** NEO 26.18.38308 (JLL, non-LTS; tag `26.18.38308.4`), xe KMD, Arc A750 (DG2, 8 GB),
Julia 1.12.6, oneAPI.jl branch `scratch_abort` (main + this doc's reproducers).
**Companion:** `SCRATCH_ABORT_ASSESSMENT.md` (Aurora LTS analysis this work probes).
**Reproducers (this checkout):** `repro78_local.jl` (host-pressure route, reconstruction of
Aurora's `repro78.jl` with `ALLOCS=1 MAPRED=1`), `vram_probe.jl` (VRAM-exhaustion route).

## Verdict

**The abort does not reproduce on the non-LTS stack on DG2/xe — but not because NEO was
fixed.** The null-check-free scratch allocation and the line-78 `UNRECOVERABLE_IF` are
byte-identical in tag `26.18.38308.4` and master (still literally line 78 of
`scratch_space_controller_xehp_and_later.cpp`). What differs is where exhaustion
manifests: under the xe KMD the scratch allocation call does not return nullptr even with
VRAM 100% full (placement is deferred/evictable under TTM), so the unguarded branch is
never entered; the shortfall surfaces at batch-buffer submission instead, which has a clean
error path (`prepareAndSubmitBatchBuffer` → `SubmissionStatus` →
`getErrorCodeForSubmissionStatus`) — in **both** 25.18-LTS and 26.18. The vulnerable
window is specific to memory managers where `allocateGraphicsMemoryWithProperties` itself
fails (PVC / i915-prelim under the LTS stack).

## Experiments

1. **Spill kernel** (`spill_kern`, 256 live accumulators): `spillMemSize=4704`,
   `privateMemSize=8192` B/thread on DG2 — more per-thread scratch than Aurora's
   `gpu_kerh2` (3648 B). Scratch demand ≈ 28 MB/slot × 2 slots at 3584 HW threads.
2. **Host-pressure route** (faithful repro78: K pinned command lists + kept 4 KiB alloc +
   `sum` mapreduce per iteration): on this 30 GB no-swap box the **host dies first**. The
   pressure loop consumes host memory *outside* the process (kernel-side, not
   memcg-charged, invisible to RSS): ~1.3 MiB/iter early, superlinear to ~5 MiB/iter past
   ~1500 pinned lists, with per-iteration time growing in step. An unconfined K=12000 run
   (≈14–17 GiB) livelocked the machine on 2026-08-10 (no OOM kill — no-swap reclaim
   thrash; power-button required). Within safe limits (K=3400, just under the watchdog
   wall at K=3691), the first spilling submission **succeeds** — host pressure alone
   cannot push the driver's internal allocation to failure without first killing a shared
   host. This superlinear kernel-side growth is a data point for the assessment's open
   item ("host-side KMD structures" as candidate exhausted resource).
3. **VRAM-exhaustion route** (`vram_probe.jl`): fill VRAM with kept USM device
   allocations (exactly 8192 MiB accepted, then clean `OutOfGPUMemoryError` at every
   granularity down to 1 MiB), then the process's first spilling submission.
   **Deterministic (2/2): `zeCommandQueueExecuteCommandLists` returns
   `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`** — the same call site that aborts on Aurora
   returns a recoverable error here. No abort, no hang, host unaffected. Notably the
   driver did *not* evict idle USM allocations to place scratch — it failed the
   submission cleanly.

## Implications for the fix plan

- The assessment's premise survives scrutiny: the abort is **latent in shared NEO code**
  (unfixed upstream as of master) but its trigger needs an eagerly-failing memory manager,
  i.e. it stays a PVC/LTS-class risk rather than a consumer-GPU one. Tier 2
  (spill-triggered drain hedge) keeps its value for those stacks; nothing observed here
  argues for more than that on current consumer drivers.
- New oneAPI.jl-side observation: on this stack, submission-time OOM is **recoverable** —
  but `execute!` does not wrap `zeCommandQueueExecuteCommandLists` in `retry_reclaim`, so
  a workload whose VRAM is mostly *unreferenced-but-unfinalized* arrays would throw
  instead of reclaiming and retrying. Wrapping the submission in `retry_reclaim` is a
  cheap, targeted improvement worth considering alongside Tier 1/2.

## Safety protocol for this machine (mandatory)

`intel-G501` is a home server (Plex/Nextcloud/Immich/HA, no swap), not a disposable test
node. Both reproducers refuse to start without a cgroup memory cap and self-abort
(rc=99) when host MemAvailable drops below `MIN_AVAIL_GIB` (default 6). Run only as:

```
systemd-run --user --scope -p MemoryMax=6G -- julia --project=test <script>
```

The cap alone is NOT sufficient — the pressure loop's kernel-side allocations are not
memcg-charged; the in-script watchdog is the guard that actually binds. rc legend:
134 abort face / 124 hang face / 99 watchdog / 137 cgroup-OOM or external SIGKILL / 2
refused-unconfined.
