using Documenter, SteadyStateFit

makedocs(;
    modules=[SteadyStateFit],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/tkf/SteadyStateFit.jl/blob/{commit}{path}#L{line}",
    sitename="SteadyStateFit.jl",
    authors="Takafumi Arakaki <aka.tkf@gmail.com>",
)

deploydocs(;
    repo="github.com/tkf/SteadyStateFit.jl",
)
