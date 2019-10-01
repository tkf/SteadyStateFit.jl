"""
    znlsolve(args...; kwargs...)

It is a Zygote-compatible version of `nlsolve(args...; kwargs...)`.

See:
https://github.com/JuliaNLSolvers/NLsolve.jl/issues/205#issuecomment-524764679
"""
znlsolve(args...; kwargs...) = NLsolve.nlsolve(args...; kwargs...)

@adjoint znlsolve(f, j, x0; kwargs...) =
    let result = znlsolve(f, j, x0; kwargs...)
        NLsolve.converged(result) ||
            throw(NLsolveNotConvergedError(f, j, x0, kwargs, result))
        result, function(vresult)
            # This backpropagator returns (- v' (df/dx)⁻¹ (df/dp))'
            v = vresult[].zero
            x = result.zero
            J = j(x)
            _, back = forward(f -> f(x), f)
            return (back(-(J' \ v))[1], nothing, nothing)
        end
    end

struct NLsolveNotConvergedError <: Exception
    f
    j
    x0
    kwargs
    result
end

function Base.showerror(io::IO, err::NLsolveNotConvergedError)
    println(io, "NLsolveNotConvergedError")
    show(io, "text/plain", err.result)
end

"""
    nlsolveargs(err::NLsolveNotConvergedError) -> (args, kwargs)

Return arguments such that `nsolve(args...; kwargs...)` would reproduce
the same result.
"""
nlsolveargs(err::NLsolveNotConvergedError) =
    ((err.f, err.j, err.x0), err.kwargs)
