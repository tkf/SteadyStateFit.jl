"""
    optimize(sso::SteadyStateObjective, [method, options, x0])

Optimize a [`SteadyStateObjective`](@ref) `sso` using Optim.  The rest
of the positional and keyword arguments are passed to
`Optim.optimize`.
"""
function optimize(
    sso::SteadyStateObjective,
    method::Optim.AbstractOptimizer = Optim.BFGS(),
    options::Optim.Options = Optim.Options(),
    x0 = get(sso.p, sso.parameterlens);
    kwargs...,
)
    return Optim.optimize(
        Optim.only_fg!(sso),
        x0,
        method,
        setupoptions(sso, options);
        kwargs...,
    )
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
