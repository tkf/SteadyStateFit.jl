"""
    solve(sso::SteadyStateObjective, [method, options, x0]) :: SteadyStateFitResult

Optimize a [`SteadyStateObjective`](@ref) `sso` using Optim.  The rest
of the positional and keyword arguments are passed to
`Optim.optimize`.
"""
function solve(
    sso::SteadyStateObjective,
    method::Optim.AbstractOptimizer = Optim.BFGS(),
    options::Optim.Options = Optim.Options(),
    x0 = get(sso.p, sso.parameterlens);
    kwargs...,
)
    result = Optim.optimize(
        Optim.only_fg!(sso),
        x0,
        method,
        setupoptions(sso, options);
        kwargs...,
    )
    return SteadyStateFitResult(result, sso)
end

"""
    SteadyStateFitResult

# Fields
- `result :: Optim.OptimizationResults`
- `objective :: SteadyStateObjective`

# Methods
- `minimum(fit::SteadyStateFitResult)`: Equivalent to `minimum(fit.result)`.
- `Optim.minimizer(fit::SteadyStateFitResult)`: Return a copy of
  `fit.objective.p` with parameters replaced by `Optim.minimizer(fit.result)`.
"""
struct SteadyStateFitResult
    result  # Optim.MultivariateOptimizationResults
    objective  # SteadyStateObjective
end

function Base.show(io::IO, ::MIME"text/plain", fit::SteadyStateFitResult)
    sso = fit.objective
    println(io, """
    SteadyStateFitResult
     * Loss:
        $(shortsummary(sso.loss; context=io))

     * Model:
        $(shortsummary(sso.p; context=io))

     * Parameters:
        $(prettylens(sso.parameterlens; context=io))

     * Conditions ($(length(sso.conditions))):
        $(prettylens(sso.conditionsetter; context=io))
    """)
    show(io, MIME("text/plain"), fit.result)
end

Base.minimum(fit::SteadyStateFitResult) = minimum(fit.result)

function Optim.minimizer(fit::SteadyStateFitResult)
    p = fit.objective.p
    parameterlens = fit.objective.parameterlens
    return set(p, parameterlens, Optim.minimizer(fit.result))
end

# A hacky solution: To use the steady state of the last iteration,
# `updatesteadystates!` is called via `callback` option.

function setupoptions(sso::SteadyStateObjective, options::Optim.Options)
    @assert options.show_every == 1
    options = @set options.extended_trace = true
    options = @set options.callback = StateUpdater(sso, options.callback)
    return options
end

struct StateUpdater{T, C}
    sso::T
    callback::C
end

function (su::StateUpdater)(os)
    # os :: OptimizationState
    updatesteadystates!(su.sso, os.metadata["x"])
    su.callback === nothing ? false : su.callback(os)
end
