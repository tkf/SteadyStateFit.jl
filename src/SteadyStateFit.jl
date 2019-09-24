module SteadyStateFit

export Optim, SteadyStateObjective, solve

import DiffEqBase: solve  # use CommonSolve.jl?
import Kaleido
import NLsolve
import Optim
using DiffEqBase: ODEProblem
using Setfield
using Zygote: @adjoint, forward

if isdefined(Kaleido, :prettylens)
    using Kaleido: prettylens
else
    prettylens(x; kwargs...) = sprint(print, x; kwargs...)
end

include("utils.jl")
include("znlsolve.jl")
include("objectives.jl")
include("solvers.jl")

end # module
