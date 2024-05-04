using CFShallowWaters
using Documenter

DocMeta.setdocmeta!(CFShallowWaters, :DocTestSetup, :(using CFShallowWaters); recursive=true)

makedocs(;
    modules=[CFShallowWaters],
    authors="The ClimFlows contributors",
    sitename="CFShallowWaters.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/CFShallowWaters.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/CFShallowWaters.jl",
    devbranch="main",
)
