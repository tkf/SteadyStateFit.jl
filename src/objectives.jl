struct SteadyStateObjective{
    TL,
    TF,
    TJ,
    TP,
    TS,
    TC,
    TCL <: Lens,
    TPL <: Lens,
}
    loss::TL   # loss(u, condition)
    f::TF      # f(u, p, t)
    j::TJ      # j(u, p, t)
    p::TP
    states::TS
    conditions::TC
    conditionsetter::TCL
    parameterlens::TPL
end

"""
    SteadyStateObjective(
        loss,
        ode::ODEProblem,
        conditions,
        conditionsetter::Lens,
        parameterlens::Lens,
    )

Solve steady states of `ode` for `N = length(conditions)` and compute
the objective using the `loss` function defined on a state-condition
pair.
"""
SteadyStateObjective(
    loss,
    ode::ODEProblem,
    conditions,
    conditionsetter,
    parameterlens,
) =
    SteadyStateObjective(
        loss, ode.f.f, ode.f.jac, ode.p,
        deepcopy(fill(ode.u0, size(conditions))),
        conditions, conditionsetter, parameterlens,
    )

setparameter(sso::SteadyStateObjective, x) =
    set(sso, (@lens _.p) ∘ sso.parameterlens, x)

updatesteadystates!(sso::SteadyStateObjective, x) =
    sso.states .= steadystates(sso, x)

steadystates(sso::SteadyStateObjective, x) = steadystates(setparameter(sso, x))

steadystates(sso::SteadyStateObjective) =
    map((condition, u0) -> _steadystate(sso, condition, u0),
        sso.states,
        sso.conditions)

_steadystate(sso::SteadyStateObjective, u0, condition) =
    let p = set(sso.p, sso.conditionsetter, condition)
        znlsolve(u -> sso.f(u, p, 0), u -> sso.j(u, p, 0), u0).zero
    end

# f(x)
(sso::SteadyStateObjective)(x) =
    let sso = setparameter(sso, x)
        sum(zip(sso.states, sso.conditions)) do (u0, condition)
            sso.loss(_steadystate(sso, u0, condition), condition)
        end
    end

# fg!(F, G, x)
#
# See:
# https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations
function (sso::SteadyStateObjective)(F, G, x)
    if G === nothing
        @assert F !== nothing  # can it happen?
        return sso(x)
    end
    f, back = forward(sso, x)
    G .= first.(back(1))
    if F !== nothing
        return f
    end
end
