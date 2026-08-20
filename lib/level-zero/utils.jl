isdebug(group) = Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug, group, oneL0) !== nothing

# Opt-out knob for the scratch hedge (`scratch_hedge!` in src/compiler/execution.jl):
# NEO's scratch-buffer allocation has no error path — on failure the process aborts —
# so the hedge drains garbage right before the first submission that triggers it.
# Initialized from ONEAPI_SCRATCH_HEDGE in `__init__`; default on.
const SCRATCH_HEDGE = Ref{Bool}(true)

# number of times the hedge actually drained; observable for tests
const SCRATCH_HEDGE_COUNT = Threads.Atomic{Int}(0)

# Registered callbacks invoked during memory reclamation (e.g., flushing deferred MKL
# sparse handle releases).  Extensions like oneMKL can register cleanup functions here
# so they run when Level Zero reports OOM or when proactive GC fires.
const _reclaim_callbacks = Function[]

function register_reclaim_callback!(f::Function)
    return push!(_reclaim_callbacks, f)
end

function _run_reclaim_callbacks()
    for cb in _reclaim_callbacks
        try
            cb()
        catch
        end
    end
    return
end

function retry_reclaim(f, isfailed)
    ret = f()

    # slow path, incrementally reclaiming more memory until we succeed
    if isfailed(ret)
        phase = 1
        while true
            if phase == 1
                GC.gc(false)
            elseif phase == 2
                GC.gc(true)
            elseif phase == 3
                # After GC, finalizers may have deferred resource releases (e.g., MKL
                # sparse handles).  Flush them now, then GC again to free the memory
                # those releases made available.
                _run_reclaim_callbacks()
                GC.gc(true)
            else
                break
            end
            phase += 1

            ret = f()
            isfailed(ret) || break
        end
    end

    ret
end
