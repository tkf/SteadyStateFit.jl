# A complicated way to do argmin x^1/2 - x^1/3
module TestExampleDiagonal

using DiffEqBase
using Setfield
using SteadyStateFit
using Test

struct NullSetter <: Lens end
Setfield.set(obj, ::NullSetter, ::Any) = obj

ode = ODEProblem(
    ODEFunction(
        (u, p, t) -> [u[1]^2, u[2]^3] .- exp.(p);
        jac = (u, p, t) -> [
            2u[1]   0
            0       3u[2]^2
        ],
    ),
    [1.0, 1.0],  # u0
    nothing,
    [1.0],  # p
)

sso = SteadyStateObjective(
    (((x1, x2),), (c,)) -> x1 - c * x2,  # loss
    ode,
    [1],                           # conditions
    NullSetter(),                  # conditionsetter
    (@lens _),                     # parameterlens
)

result = solve(sso)

@test exp(Optim.minimizer(result)[1]) ≈ 64/729  rtol=0.01
@test occursin("SteadyStateFitResult", sprint(show, "text/plain", result))

result2 = solve(sso, Optim.BFGS(), Optim.Options(store_trace=true))
@test exp(Optim.minimizer(result2)[1]) ≈ 64/729  rtol=0.01

end  # module
