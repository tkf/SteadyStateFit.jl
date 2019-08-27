"""
    znlsolve(args...; kwargs...)

It is a Zygote-compatible version of `nlsolve(args...; kwargs...)`.

See:
https://github.com/JuliaNLSolvers/NLsolve.jl/issues/205#issuecomment-524764679
"""
znlsolve(args...; kwargs...) = NLsolve.nlsolve(args...; kwargs...)

@adjoint znlsolve(f, j, x0; kwargs...) =
    let result = znlsolve(f, j, x0; kwargs...)
        NLsolve.converged(result) || throw(NLsolveNotConvergedError(result))
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
    result
end

function Base.showerror(io::IO, err::NLsolveNotConvergedError)
    println(io, "NLsolveNotConvergedError")
    show(io, "text/plain", err.result)
end
