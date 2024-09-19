#======== All =========#

scratch_space(model, state) = scratch_SW(model.domain, state)
tendencies!(dstate, model, state, scratch, t) = tendencies_SW!(dstate, state, scratch, model, model.domain)

#========= Voronoi ========#

using CFDomains: Stencils

function same(x,y)
    @assert x==y
    return x
end

function scratch_SW(domain::VoronoiSphere, (; ucov, ghcov))
    F = same(eltype(ucov), eltype(ghcov))
    allocate_fields((qv=:dual, qe=:vector, U=:vector, B=:scalar), domain, F)
end

function tendencies_SW!( dstate, (; ucov,ghcov), scratch, model, mesh::VoronoiSphere)
    radius = model.planet.radius
    U = massflux!(scratch.U, ucov, ghcov, radius, mesh)
    B = bernoulli!(scratch.B, ghcov, ucov, radius, mesh)
    qv = voronoi_potential_vorticity!(scratch.qv, model.fcov, ucov, ghcov, mesh)
    ducov = voronoi_du!(dstate.ucov, scratch.qe, qv, U, B, mesh)
    dghcov = voronoi_dm!(dstate.ghcov, U, mesh)
    return (ghcov=dghcov, ucov=ducov)
end

similar!(::Void, x...) = similar(x...)
similar!(y, x...) = y

#== Voronoi, mass flux ==#

function massflux!(U_, ucov, m, radius, vsphere)
    U = similar!(U_, ucov)
    @fast for ij in eachindex(U)
        flux = Stencils.centered_flux(vsphere, ij)
        U[ij] = inv(radius*radius)*flux(m, ucov) # scale by contravariant metric
    end
    return U
end

#== Voronoi, kinetic energy ==#

function bernoulli!(B_, gh, ucov, radius, vsphere)
    B = similar!(B_, gh)
    inv_r2 = inv(radius*radius)
    @fast for ij in eachindex(B)
        deg = vsphere.primal_deg[ij]
        @unroll deg in 5:7 begin
            dot_product = Stencils.dot_product(vsphere, ij, Val(deg))
            B[ij] = inv_r2 * (gh[ij] + dot_product(ucov, ucov)/2)
        end
    end
    return B
end

#=========== Fluid Poisson bracket elements ===========#

#== Voronoi, potential vorticity q = curl(ucov)/m ==#

function voronoi_potential_vorticity!(qv_, fv, ucov, m, vsphere)
    qv = similar!(qv_, fv, eltype(ucov))
    @fast for ij in eachindex(qv)
        zeta = Stencils.curl(vsphere, ij)(ucov)
        mv = vsphere.Av[ij]*Stencils.average_iv(vsphere, ij)(m)
        qv[ij] = inv(mv)*(fv[ij]+zeta)
    end
    return qv
end

#== mass tendency dm = -div(U) ==#

voronoi_dm!(::Void, U, areas, degree, edges, signs) =
    voronoi_dm!(similar(areas, eltype(U)), U, areas, degree, edges, signs)

function voronoi_dm!(dm_, U, vsphere)
    dm = similar!(dm_, vsphere.Ai, eltype(U))
    @fast for ij in eachindex(dm)
        deg = vsphere.primal_deg[ij]
        @unroll deg in 5:7 begin
            div = Stencils.divergence(vsphere, ij, Val(deg))
            dm[ij] = -div(U)
        end
    end
    return dm
end

#== velocity tendency du = -q x U - grad B ==#

function voronoi_du!(du_, qe_, qv, U, B, mesh)
    qe = similar!(qe_, U)
    du = similar!(du_, U)
    # interpolate PV q from v-points (dual cells=triangles) to e-points (edges)
    @fast for ij in eachindex(qe)
        qe[ij] = Stencils.average_ve(mesh, ij)(qv)
    end
    @fast for ij in eachindex(du)
        grad = Stencils.gradient(mesh, ij)
        deg = mesh.trisk_deg[ij]
        @unroll deg in 9:12 begin
            trisk = Stencils.TRiSK(mesh, ij, Val(deg))
            du[ij] = trisk(U, qe) - grad(B)
        end
    end
    return du
end
