"""
    znlsolve(args...; kwargs...)

It is a Zygote-compatible version of `nlsolve(args...; kwargs...)`.

See:
https://github.com/JuliaNLSolvers/NLsolve.jl/issues/205#issuecomment-524764679
"""
znlsolve(args...; converged=nothing, adlinsolve=nothing, kwargs...) =
    NLsolve.nlsolve(args...; kwargs...)

@adjoint znlsolve(
    f,
    j,
    x0;
    converged = NLsolve.converged,
    adlinsolve = LinSolvePinvFallback(eltype(x0)),
    kwargs...,
) =
    let result = znlsolve(f, j, x0; kwargs...)
        converged(result) ||
            throw(NLsolveNotConvergedError(f, j, x0, kwargs, result))
        result, function(vresult)
            # This backpropagator returns (- v' (df/dx)⁻¹ (df/dp))'
            v = vresult[].zero
            x = result.zero
            J = j(x)
            _, back = forward(f -> f(x), f)
            \ = adlinsolve
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

struct LinSolvePinvFallback{T <: Real}
    det_threshold::T
    pinv_atol::T
    pinv_rtol::T
end

LinSolvePinvFallback(
    T::Type = Float64;
    det_threshold = sqrt(eps(T)),
    pinv_atol = zero(T),
    pinv_rtol = sqrt(eps(T)),
) = LinSolvePinvFallback{T}(
    det_threshold,
    pinv_atol,
    pinv_rtol,
)

function (linsolve::LinSolvePinvFallback)(A, B)
    F = lu(A)
    if abs(det(F)) > linsolve.det_threshold
        return F \ B
    end
    return pinv(A; atol=linsolve.pinv_atol, rtol=linsolve.pinv_rtol) * B
end
