module Voronoi

using ManagedLoops: @loops, @unroll
using MutatingOrNot: similar!, has_dryrun

using CFDomains: VoronoiSphere, allocate_fields
using CFDomains: VoronoiOperators as Ops
using CFDomains.LazyExpressions: @lazy

using MutatingOrNot: void, Void
using CFShallowWaters: @fast

import CFShallowWaters: scratch_SW, tendencies_SW!

function same(x,y)
    @assert x==y
    return x
end

function scratch_SW(domain::VoronoiSphere, (; ucov, ghcov))
    F = same(eltype(ucov), eltype(ghcov))
    allocate_fields((ghv=:dual, zetav=:dual, qe=:vector, U=:vector, u2=:scalar), domain, F)
end

function tendencies_SW!(dstate, scratch, (; ghcov, ucov), model, mesh::VoronoiSphere)
    inv_Ai, fcov, radius = mesh.inv_Ai, model.fcov, model.planet.radius
    metric = radius^-2

    dghcov = similar!(dstate.ghcov, ghcov)
    ducov = similar!(dstate.ucov, ucov)

    U = similar!(scratch.U, ucov)
    u2 = similar!(scratch.u2, ghcov)
    zetav = similar!(scratch.zetav, fcov)
    ghv = similar!(scratch.ghv, fcov)
    qe = similar!(scratch.qe, ucov)

    result = (; ghcov=dghcov, ucov=ducov), (; U, u2, ghv, zetav, qe)
    has_dryrun(dstate, scratch) && return result # early exit if only allocation desired

    cflux! = Ops.CenteredFlux(mesh)
    square! = Ops.SquaredCovector(mesh)
    curl! = Ops.Curl(mesh) # 1-form -> 2-form
    to_dual! = Ops.DualFromPrimal(mesh) # 0-form -> 2-form
    to_edge! = Ops.EdgeFromDual(mesh) # 0-form -> 0-form
    minus_div! = Ops.Divergence(mesh, Ops.setminus!)
    minus_grad! = Ops.Gradient(mesh, Ops.setminus!) # 0-form -> 1-form
    add_trisk! = Ops.EnergyTRiSK(mesh, Ops.addto!) # (2-form, 0-form at edges) -> 1-form

    @lazy ucontra(ucov ; metric) = metric*ucov
    @lazy gh0(ghcov ; inv_Ai) = inv_Ai*ghcov
    cflux!(U, nothing, gh0, ucontra)
    minus_div!(dghcov, nothing, U)

    square!(u2, nothing, ucov)
    @lazy B(u2, ghcov ; inv_Ai, metric) = (metric*inv_Ai)*(u2/2 + ghcov)
    minus_grad!(ducov, nothing, B)

    to_dual!(ghv, nothing, gh0)
    curl!(zetav, nothing, ucov)
    @lazy qv(zetav, ghv ; fcov) = (zetav+fcov)/ghv
    to_edge!(qe, nothing, qv)
    add_trisk!(ducov, nothing, U, qe)

    return result
end

end # module Voronoi
