"""
    solve(sso::SteadyStateObjective, [method, options, x0])

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
    iter = Optim.optimizing(
        Optim.only_fg!(sso),
        x0,
        method,
        options;
        kwargs...,
    )
    return Optim.OptimizationResults(_solve!(iter, sso))
end

function _solve!(iter, sso)
    local istate
    for istate′ in iter
        istate = istate′
        updatesteadystates!(sso, Optim.minimizer(istate))
    end
    return istate
end
