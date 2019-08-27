var documenterSearchIndex = {"docs":
[{"location":"#SteadyStateFit.jl-1","page":"Home","title":"SteadyStateFit.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"SteadyStateFit.SteadyStateObjective\nSteadyStateFit.optimize","category":"page"},{"location":"#SteadyStateFit.SteadyStateObjective","page":"Home","title":"SteadyStateFit.SteadyStateObjective","text":"SteadyStateObjective(\n    loss,\n    ode::ODEProblem,\n    conditions,\n    conditionsetter::Lens,\n    parameterlens::Lens,\n)\n\nSteadyStateObjective defines an objective function F\n\nF(x) =\nfrac1N\nsum_c in mathttconditions mathttloss(u(x c) c)\n\nwhere N = length(conditions) and u(x c) is the steady state solution of the ode given a trainable parameter x and a condition c.  The trainable parameter x and the \"external\" condition c are set using parameterlens and conditionsetter lenses which act on ode.p respectively.\n\nSteadyStateObjective also provides Jacobian in the fg!(F, G, x) form required by Optim.only_fg!.\n\n\n\n\n\n","category":"type"},{"location":"#SteadyStateFit.optimize","page":"Home","title":"SteadyStateFit.optimize","text":"optimize(sso::SteadyStateObjective, [x0, method, options])\n\nOptimize a SteadyStateObjective sso using Optim.  The rest of the positional and keyword arguments are passed to Optim.optimize.\n\n\n\n\n\n","category":"function"}]
}
