module SteadyStateFit

export Optim, SteadyStateObjective, solve

import DiffEqBase: solve  # use CommonSolve.jl?
import NLsolve
import Optim
using DiffEqBase: ODEProblem
using Setfield
using Zygote: @adjoint, forward

include("znlsolve.jl")
include("objectives.jl")
include("solvers.jl")

end # module
