# The NEO scratch abort: root cause, assessment, and fix plan for oneAPI.jl

**Date:** 2026-08-07 (session on Aurora x4310c1s1b0n0, PBS job 8742234).
**Stack:** intel-compute-runtime 25.18.33578.42 (Aurora LTS), oneAPI.jl branch
`cache-launch-configuration` at `5f74421`, Julia 1.12.6, Max 1550, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`.
**Companion documents:** `~/../exa-models-paper/benchmark/PERF_LAUNCH_OVERHEAD.md`
("2026-08-07" section) has the full experiment history; reproducers and logs live in
`exa-models-paper/benchmark/data/perf-scratch-20260806/` (`repro78.jl`, `exa_keep.jl`,
`exa_pretouch.jl`, `logs-abort/`).
**Decision context:** no Intel bug report for now — the failure has only been reproduced on
the LTS stack, which is outdated. The fix must be general to oneAPI.jl, not a
benchmark-harness workaround.

---

## 1. What the abort is

The recurring failure signature, seen only in workloads containing ExaModels' `shessian!`
(kernel `gpu_kerh2`, the only kernel in the workload with non-zero spill, 3648 B/thread):

```
Abort was called at 78 line in file:
.../shared/source/command_stream/scratch_space_controller_xehp_and_later.cpp
signal (6): Aborted
  zeCommandQueueExecuteCommandLists
```

Line 78 at the exact driver tag (fetched from GitHub, `25.18.33578.42`), in
`ScratchSpaceControllerXeHPAndLater::programSurfaceState()`:

```cpp
UNRECOVERABLE_IF(scratchSlot0Allocation == nullptr && scratchSlot1Allocation == nullptr);
```

The only route to it: `prepareScratchAllocation()` decides the submission needs more
scratch than allocated, sets `scratchSurfaceDirty = true`, then does

```cpp
scratchSlot0Allocation = getMemoryManager()->allocateGraphicsMemoryWithProperties(properties);
```

**with no null check**. If that allocation fails, `programSurfaceState()` runs anyway and
the process is aborted. So the abort is precisely: **NEO failed to allocate the scratch
buffer and has no error path for it.** Resource exhaustion — not a race, not memory
corruption, not a GC-rooting bug (all previously ruled out by measurement).

Two structural corollaries, both verified against the source and consistent with every
observation:

- **Scratch is allocated once per queue, at the first spilling submission**, and only
  re-triggers if a later kernel needs *more* scratch. A process whose first spilling
  submission happens while memory is pristine is immune for the rest of its life.
  This retroactively explains every "reproducer that never reproduced":
  `repro_spill_abort.jl` launched the spill kernel once at startup "to get compilation out
  of the way" — which allocates scratch cleanly and inoculates the process. (Its header
  still claims the abort was a GC-rooting bug fixed by `GC.@preserve`; that claim was
  withdrawn on 2026-08-06 and is superseded by this document.)
- **The intermittency is a GC lottery.** What matters is how much driver-side memory is
  live at the single instant the scratch allocation happens; that depends on when the GC
  last ran finalizers. Every change that made the host faster made that instant more
  crowded, which is why "every fast configuration aborts and the slow ones don't" was the
  observed pattern.

## 2. Three faces of one wall

The same exhaustion produces three different symptoms depending on which allocation hits
the wall first and whether that path has error handling:

| face | mechanism | observed |
|---|---|---|
| **abort** (rc=134) | NEO-internal scratch allocation fails; no error path → line 78 | 12 logged runs, 2026-08-05/06 |
| **hang** (rc=124) | submit/append blocks in an `ioctl` inside the driver, or returns OOM into `retry_reclaim` which cannot free enough; two runs outlived SIGTERM by 2–10 min | 6 runs, 2026-08-07 |
| **clean** (rc=0) | GC happened to keep the population low at the critical moments | interleaved with both |

**Any rate measurement must count rc=124 and rc=134 as the same event.**

Key experiments behind this (2026-08-07):

- `exa_keep.jl` — the real workload with **every `ZeCommandList` pinned** (finalizers can
  never destroy one): hangs in `retry_reclaim` at `zeCommandListAppendLaunchKernel`, at the
  same workload point where the aborts happen (first model's hess warm-up). With pinned
  lists GC can free nothing, so the OOM is persistent → hang instead of recovery.
- Unmodified baseline, same build, same tile: 1/4 failed — as the **hang** face, blocked in
  an `ioctl` inside `zeCommandQueueExecuteCommandLists` for minutes.
- `repro78.jl` — pure oneAPI.jl (no ExaModels): K no-spill launches with all lists pinned,
  **then** the process's first spilling submission. Survived at every K up to 24000.
  **Command lists alone are not the pressure.**
- `repro78.jl` with `ALLOCS=1 MAPRED=1` (one kept 4 KiB `oneAPI.zeros` + one `sum(a)`
  mapreduce per launch, mimicking what real callbacks allocate): hit the wall in the
  pressure phase at K=12000 — same blocked-in-ioctl signature as the real workload,
  without ExaModels. **Device-array / USM churn is a necessary ingredient.**
- Cross-tile interference: every failure today co-occurred with a pressure experiment
  running on a *different tile* of the same node. Part of the wall is shared KMD/host
  driver state, not per-tile device memory. Rates are only meaningful from solo runs.

## 3. Why oneAPI.jl is exposed

`execute!` (`lib/level-zero/cmdlist.jl:111`) creates a fresh `ZeCommandList` per kernel
dispatch and **drops its only reference on return**. Destruction is entirely at the mercy
of finalizer timing. Under a launch storm, thousands of driver objects (command lists with
command buffers and heaps) plus un-finalized `oneArray`s / mapreduce temporaries accumulate
between GC cycles. The driver's own allocations then land on a nearly-full system, and the
one allocation with no error path — scratch — aborts the process.

CUDA.jl and AMDGPU.jl push work onto streams and create no per-dispatch objects, which is
presumably why this failure class does not exist there.

## 4. Fix plan

Three tiers, in increasing ambition. File anchors are for this checkout
(`/lus/flare/projects/Julia/mschanen/git/oneAPI.jl`).

### Tier 1 — bounded pending-list drain (workaround, ~50 lines, low risk)

Make command-list lifetime deterministic instead of GC-driven:

- Keep every submitted list in per-queue storage instead of dropping it. The queue
  registry (`register_queue!` / `synchronize_all_queues`) is the natural place to hang it.
- Drain — queue-synchronize, then explicitly `Base.finalize` the retired lists — at three
  trigger points:
  1. whenever `synchronize()` runs anyway (free);
  2. when the pending count crosses a cap (512–1024), so a sync-free launch storm can
     never accumulate more than the cap;
  3. from a `register_reclaim_callback!` hook (`lib/level-zero/utils.jl`), so
     `retry_reclaim` frees driver objects deterministically under OOM instead of hoping
     finalizers run.

Ordering matters: **syncing alone is not enough** — `ONEAPI_SYNC_EACH_SUBMISSION=1` still
aborted (measured 2026-08-06), because syncing retires work but does not destroy lists.
The drain must actually finalize them.

Semantics-preserving, benefits every consumer on every driver. This is insurance in case
Tier 3 stalls; if Tier 3 lands, this bookkeeping is deleted.

### Tier 2 — spill-triggered drain hedge (~15 lines, keep under any architecture)

The dangerous moment is the first submission of a kernel whose `spillMemSize` exceeds the
queue's scratch high-water mark — that is when NEO performs the null-check-free
allocation. The launch path already has the kernel's properties (the per-kernel
`launch_configuration` cache carries them, and the spill-aware group-size commit
`8382cda` already reads `spillMemSize`), so:

- track the maximum spill seen per queue;
- when a kernel exceeds it, drain pending lists + `GC.gc(false)` once, **then** submit.

Fires once per (queue, scratch tier); costs one sync; makes the fatal allocation happen at
the cleanest moment available rather than the dirtiest. This is the generalization of the
"pre-touch" idea, but living in oneAPI.jl and requiring nothing from users.

Note this is the **only** tier that specifically guards the scratch allocation, and the
missing null check is in NEO's shared code — nothing suggests it is LTS-specific, so this
hedge has value on current drivers too.

### Tier 3 — immediate command lists (the real fix, and the performance fix)

One `zeCommandListCreateImmediate` per (context, device, queue) in ASYNCHRONOUS mode;
append launches (and copies — `copyto!` also goes through `execute!`) directly. The
per-dispatch list object — the garbage source — disappears entirely, and per-dispatch cost
drops from 1.13e-5 s to 1.41e-6 s (8x, measured in `cmdlist.jl` bench, output verified).
Bindings exist (`lib/level-zero/libze.jl:2484`); nothing wraps them.

Design surface (why this is the medium-term item, not the afternoon patch):

- `synchronize` semantics: `zeCommandListHostSynchronize` (L0 ≥ 1.6) or an event/fence per
  batch instead of queue sync; `register_queue!`/`synchronize_all_queues` needs rethinking.
- The LTS `sync_each_submission` workaround must be re-expressed (likely trivially:
  host-synchronize after each append).
- Interaction with events and multi-queue use.
- Validation across drivers (Arc, integrated, PVC LTS vs current NEO) on hardware that, for
  PVC, is only reachable in 1-hour debug allocations.

Fallback if immediate lists interact badly with the LTS stack: pool and reuse regular
lists per queue (reset instead of create/destroy; 5.2x, measured) — which is Tier-1
bookkeeping plus reuse.

## 5. Why not only Tier 3

Two genuine gaps, one empirical and one practical:

1. **Immediate lists do not close the abort by themselves.** The pure sweep is the
   evidence: 24k pinned command lists — the worst possible list pressure — and the first
   spilling submission still succeeded; the wall only appeared once device-array churn was
   added. Un-finalized `oneArray`s and mapreduce temporaries are co-conspirators, they are
   still GC-reclaimed under Tier 3, and NEO's scratch path still aborts rather than errors.
   Rarer, but reachable. Tier 2 is the only targeted guard for that allocation.
2. **Risk/validation asymmetry.** Tier 3 changes submission and synchronization semantics
   for every consumer and needs hardware validation across driver generations; Tiers 1–2
   are additive and semantics-preserving, testable in one debug job. If Tier 3 hits an LTS
   blocker, the fallback is Tier-1 bookkeeping anyway.

**Coherent minimal-work position if committed to Tier 3: do 3, keep 2, skip 1.** Tier 2
is not scaffolding that 3 obsoletes — it guards a different resource (scratch) against a
different garbage source (USM churn). Tier 1 is the droppable piece; decide after a first
immediate-list prototype, not before.

Sequencing note: Tier 3 is also the performance fix, so it perturbs every measured number.
If a near-term Intel re-measurement should happen on something upstreamable, Tier 2 alone
on the current branch gives abort-safety without changing the timing story; Tier 3 then
lands as its own PR with its own numbers.

## 6. Validation protocol (for whichever tier is implemented)

- Run the rate harness **solo on the node** — no concurrent experiments on other tiles
  (cross-tile interference is measured and real).
- Alternate fixed/unfixed builds on the same tile, N ≥ 8 runs each.
- Count **any rc ≠ 0** as a failure (abort 134 and hang 124 are the same event).
- Reproducers: `exa_noinstr.jl` (real workload), `repro78.jl` with `ALLOCS=1 MAPRED=1`
  (pure oneAPI.jl). For the line-78 face specifically, the untested-but-designed probe is
  churn with K just below the phase-A wall (K = 3000–6000), so the first failing
  allocation is phase B's scratch.

## 7. Open items

- The exact resource being exhausted is still unidentified (not list count alone; the
  churn ingredient is necessary). Candidates: a 4 GB driver heap for command buffers /
  heaps, BO-handle or residency-list growth, host-side KMD structures. Identifying it is
  only needed for an eventual Intel report, not for the oneAPI.jl fix.
- Two runs survived SIGTERM for minutes inside driver ioctls — a second driver defect
  (blocking instead of failing), worth including if a report is ever filed.
- Fix the stale header of `repro_spill_abort.jl` (this checkout) which still attributes
  the abort to a GC-rooting bug.
