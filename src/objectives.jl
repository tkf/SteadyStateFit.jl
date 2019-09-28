struct SteadyStateObjective{
    TL,
    TF,
    TJ,
    TP,
    TS,
    TC,
    TCL <: Lens,
    TPL <: Lens,
    TSSS,
}
    loss::TL   # loss(u, condition, sso)
    f::TF      # f(u, p, t)
    j::TJ      # j(u, p, t)
    p::TP
    states::TS
    conditions::TC
    conditionsetter::TCL
    parameterlens::TPL
    steadystatesolver::TSSS
end

struct NLSolver{T}
    options::T
end

NLSolver(; kwargs...) = NLSolver(kwargs.data :: NamedTuple)

"""
    SteadyStateObjective(
        loss,
        ode::ODEProblem,
        conditions::AbstractVector,
        conditionsetter::Lens,
        parameterlens::Lens,
        steadystatesolver = NLSolver(),
    )

`SteadyStateObjective` defines an objective function that `x ↦ F(x)` that
computes

```julia
loss(states :: AbstractVector{TU}, conditions :: AbstractVector{TC}, sso) :: Real
```

where

* `states` is a vector of steady state `u(x, c :: TC) :: TU` of the
  `ode` computed given a trainable parameter `x` and each `c` in
  `conditions`,

* `conditions` is the argument passed to `SteadyStateObjective`, and

* `sso` is a `SteadyStateObjective` such that `sso.p` contains the
  parameter `x` for which the objective is evaluated; i.e. `sso.p` is
  `set(ode.p, parameterlens, x)`.

The trainable parameter `x` and the "external" condition `c` are set
using `parameterlens` and `conditionsetter` lenses which act on
`ode.p` respectively.

`SteadyStateObjective` also provides Jacobian in the `fg!(F, G, x)`
form required by `Optim.only_fg!`.
"""
SteadyStateObjective(
    loss,
    ode::ODEProblem,
    conditions,
    conditionsetter,
    parameterlens,
    steadystatesolver = NLSolver(),
) =
    SteadyStateObjective(
        loss, ode.f.f, ode.f.jac, ode.p,
        deepcopy(fill(ode.u0, size(conditions))),
        conditions, conditionsetter, parameterlens,
        steadystatesolver,
    )

setparameter(sso::SteadyStateObjective, x) =
    set(sso, (@lens _.p) ∘ sso.parameterlens, x)

updatesteadystates!(sso::SteadyStateObjective, x) =
    sso.states .= steadystates(sso, x)

setsteadystates!(f, sso::SteadyStateObjective) =
    foreach(sso.states, sso.conditions) do u0, condition
        u0 .= f(set(sso.p, sso.conditionsetter, condition), u0)
    end

"""
    steadystates(sso::SteadyStateObjective, [x])

Compute steady states.  Use the (initial) parameter set for `sso.p` unless
the parameter value `x` is given.
"""
steadystates(sso::SteadyStateObjective, x) = steadystates(setparameter(sso, x))

steadystates(sso::SteadyStateObjective) =
    map((u0, condition) -> _steadystate(sso, u0, condition),
        sso.states,
        sso.conditions)

_steadystate(sso::SteadyStateObjective, u0, condition) =
    let p = set(sso.p, sso.conditionsetter, condition)
        # TODO: dispatch on `steadystatesolver` type at `znlsolve`
        options = sso.steadystatesolver.options
        znlsolve(u -> sso.f(u, p, 0), u -> sso.j(u, p, 0), u0; options...).zero
    end

# f(x)
(sso::SteadyStateObjective)(x) =
    let sso = setparameter(sso, x)
        states = map(sso.states, sso.conditions) do u0, condition
            _steadystate(sso, u0, condition)
        end
        sso.loss(states, sso.conditions, sso)
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
    # Should I warn when the gradient is `nothing`?
    G .= something.(first(back(1)), false)
    if F !== nothing
        return f
    end
end

function _showbody(io, sso::SteadyStateObjective)
    print(io, """
     * Loss:
        $(shortsummary(sso.loss; context=io))

     * Model:
        $(shortsummary(sso.p; context=io))

     * Parameters:
        $(prettylens(sso.parameterlens; context=io))

     * Conditions ($(length(sso.conditions))):
        $(prettylens(sso.conditionsetter; context=io))
    """)
end

function Base.show(io::IO, ::MIME"text/plain", sso::SteadyStateObjective)
    println(io, "SteadyStateObjective")
    _showbody(io, sso)
end
