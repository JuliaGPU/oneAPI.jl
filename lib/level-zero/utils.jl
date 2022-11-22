isdebug(group) = Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug, group, oneL0) !== nothing

macro retry_reclaim(isfailed, ex)
    quote
        ret = $(esc(ex))

        # slow path, incrementally reclaiming more memory until we succeed
        if $(esc(isfailed))(ret)
            phase = 1
            while true
                if phase == 1
                    GC.gc(false)
                elseif phase == 2
                    GC.gc(true)
                else
                    break
                end
                phase += 1

                ret = $(esc(ex))
                $(esc(isfailed))(ret) || break
            end
        end

        ret
    end
end
