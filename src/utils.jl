function maybe_default_show(io, x)
    if !get(io, :limit, false)
        invoke(show, Tuple{IO, Any}, io, x)
        return true
    end
    return false
end

function shortsummary(x; context=devnull)
    context = IOContext(
        context,
        :compact => true,
        :limit => true,
    )
    return first(sort(
        [
            sprint(summary, x; context=context),
            sprint(print, x; context=context),
        ],
        by = length,
    ))
end

macro nograd(funs...)
    exprs = map(funs) do f
        f = esc(f)
        args = esc(gensym("args"))
        return macroexpand(
            __module__,
            :(@adjoint $f($args...) = $f($args...), _ -> nothing),
        )
        # Without `macroexpand`, I get `syntax:
        # "SteadyStateFit.__context__" is not a valid function
        # argument name`.
    end
    return Expr(:block, __source__, exprs...)
end

# https://github.com/FluxML/Zygote.jl/pull/351
@nograd Base.gc_num Base.time_ns

nograd(f) = f()
@nograd nograd
