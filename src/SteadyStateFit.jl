module SteadyStateFit

export Optim, SteadyStateObjective, solve

import DiffEqBase: solve  # use CommonSolve.jl?
import Kaleido
import NLsolve
import Optim
using DiffEqBase: ODEProblem
using LinearAlgebra: lu, pinv, det
using Setfield
using ZygoteRules: @adjoint

if isdefined(Kaleido, :prettylens)
    using Kaleido: prettylens
else
    prettylens(x; kwargs...) = sprint(print, x; kwargs...)
end

include("utils.jl")
include("znlsolve.jl")
include("objectives.jl")
include("solvers.jl")

function __init__()
    # Load Zygote only at run-time:
    if ccall(:jl_generating_output, Cint, ()) != 1
        Zygote = Base.require(Base.PkgId(
            Base.UUID("e88e6eb3-aa80-5325-afca-941959d7151f"),
            "Zygote",
        ))
        @eval const forward = $Zygote.forward
    end
end

end # module
