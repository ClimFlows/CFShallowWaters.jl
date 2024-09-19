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
    ducov = voronoi_du!(dstate.ucov, scratch.qe, qv, U, B, mesh.edge_down_up,
        mesh.edge_left_right, mesh.trisk_deg, mesh.trisk, mesh.wee)
    dghcov = voronoi_dm!(dstate.ghcov, U, mesh)
    return (ghcov=dghcov, ucov=ducov)
end

function tendencies_SW( (; ucov, ghcov), model, mesh::VoronoiSphere)
    radius = model.planet.radius
    U = massflux!(void, ucov, ghcov, radius, mesh)
    B = bernoulli(ghcov, ucov, radius, mesh.primal_deg, mesh.Ai, hodges, mesh.primal_edge)
    qv = voronoi_potential_vorticity!(void, model.fcov, ucov, ghcov, mesh.Av, mesh.dual_vertex, mesh.dual_edge, mesh.dual_ne, mesh.Riv2)
    ducov = voronoi_du!(void, qv, U, B, mesh.edge_down_up,
        mesh.edge_left_right, mesh.trisk_deg, mesh.trisk, mesh.wee)
    dghcov = voronoi_dm!(void, U, mesh.Ai, mesh.primal_deg, mesh.primal_edge, mesh.primal_ne)
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
        @unroll deg in 5:7 dm[ij] = -Stencils.divergence(vsphere, ij, Val(deg))(U)
    end
    return dm
end

#== velocity tendency du = -q x U - grad B ==#

voronoi_du!(::Void, ::Void, qv, U, B, down_up, left_right, degree, edges, w) =
    voronoi_du!(similar(U), similar(U), qv, U, B, down_up, left_right, degree, edges, w)

function voronoi_du!(du::V, qe::V, qv, U, B, down_up, left_right, degree, edges, w) where {V<:AbstractVector}
    # interpolate PV q from v-points (dual cells=triangles) to e-points (edges)
    @fast for ij in eachindex(qe)
        qe[ij] = (1//2)*(qv[down_up[2,ij]]+qv[down_up[1,ij]])
    end
    @fast for ij in eachindex(du)
        deg=degree[ij]
        @unroll deg in 9:12 qV = (1//2)*sum(
            U[edges[e,ij]]*(qe[ij]+qe[edges[e,ij]])*w[e,ij] for e in 1:deg )
        # Remark : the sign convention of wee is such that
        # du = dB + qV, not du = dB-qV
        du[ij] = (B[left_right[1,ij]]-B[left_right[2,ij]]) + qV
    end
    return du
end

voronoi_du!(du::AbstractMatrix, args...) = voronoi_du_3D!(threaded, voronoi_du_wUq_vec!, du, args...)

function voronoi_du_3D!(backend, wUq!::Fun, du, qe, qv, U, B, down_up, left_right, degree, edges, w, N) where Fun
    offload(voronoi_du_qe!, backend, axes(qe), qe, qv, down_up)

    barrier(backend, @__LINE__)

    offload(wUq!, backend, axes(du),
#    offload(wUq!, backend, axes(du,2), axes(du,1),
        du, qe, U, B, left_right, degree, edges, w, N)
    return du
end

# Interpolates PV q from v-points (dual cells=triangles) to e-points (edges)
function voronoi_du_qe!((krange, ijrange), qe,qv, down_up)
    @fast for ij in ijrange
        down, up = down_up[1,ij], down_up[2,ij]
        @simd for k in krange
            qe[k,ij] = half(qv[k,up]+qv[k,down])
        end
    end
end

# The following two functions are two slightly different implementations of du = -q x U - grad(B)
# where q has been interpolated to e-points (edges) by voronoi_du_qe!.
# One of these functions is passed to voronoi_du_3D! as the argument wUq!

# In this implementation, the local loop over edges is not unrolled. Instead it is
# made the outer loop, so that the inner loop is w.r.t the vertical index, and vectorizes.
@inline function voronoi_du_wUq_vec!((krange, ijrange)::Tuple, du, qe,U,B, left_right, degree, edges, w, N)
    voronoi_du_wUq_vec!(ijrange, krange, du, qe,U,B, left_right, degree, edges, w, N)
end

function voronoi_du_wUq_vec!(ijrange, krange, du, qe,U,B, left_right, degree, edges, w, N)
    @fast for ij in ijrange
        for k in krange
            du[k,ij] = 0
        end
        deg = degree[ij]
        for e = 1:deg
            ww = w[e,ij]
            ee = edges[e,ij]
            @simd for k in krange
                du[k,ij] = muladd(ww*U[k,ee], qe[k,ij]+qe[k,ee], du[k,ij])
            end
        end
        voronoi_du_grad!(ij, krange, du, B, left_right, N)
    end
end

# In this implementation, the loop over vertical indices is the outer loop
# The inner loop over edges is unrolled in batches of 3 edges
# Indeed full unrolling prevents vectorization
function voronoi_du_wUq_unroll!(range, tag, du, qe,U,B, left_right, degree, edges, w, N)
    nz = size(du,1)
    @fast for ij in range
        @simd for k=1:nz
            du[k,ij] = 0
        end
        @unroll for batch=1:3
            off = 3*(batch-1)
            ww = ( w[e+off,ij] for e in 1:3 )
            ee = ( edges[e+off,ij] for e in 1:3 )
            @simd for k=1:nz
                du[k,ij] += sum( (ww[e]*U[k,ee[e]])*(qe[k,ij]+qe[k,ee[e]]) for e in 1:3 )
            end
        end
        # finish the sum if more than 9 edges
        off = 9
        deg = degree[ij]-off
        @unroll deg in 1:3 begin # number of edges = deg+off
            ww = ( w[e+off,ij] for e in 1:deg )
            ee = ( edges[e+off,ij] for e in 1:deg )
            @simd for k=1:nz
                du[k,ij] += sum( (ww[e]*U[k,ee[e]])*(qe[k,ij]+qe[k,ee[e]]) for e in 1:deg )
            end
        end
        voronoi_du_grad!(ij, du, B, left_right, N)
    end
end

@inline function voronoi_du_grad!(ij, krange, du, B::AbstractMatrix, left_right, ::Val{0})
    @fast begin
        left, right = left_right[1,ij], left_right[2,ij]
        @simd for k in krange
            # Remark : the sign convention of wee is such that
            # du = dB + qV, not du = dB-qV
            du[k,ij] = (B[k,left]-B[k,right]) + (1//2)*du[k,ij]
        end
    end
end

@inline function voronoi_du_grad!(ij, krange, du, B::AbstractArray, left_right, ::Val{1})
    @fast begin
        theta = view(B,:,:,2)
        exner = view(B,:,:,3)
        B     = view(B,:,:,1)
        left, right = left_right[1,ij], left_right[2,ij]
        @simd for k in krange
            # Remark : the sign convention of wee is such that
            # du = dB + qV, not du = dB-qV
            dB = B[k,left]-B[k,right]
            dExner = exner[k,left]-exner[k,right]
            theta2 = theta[k,left]+theta[k,right]
            du[k,ij] = dB + half(theta2*dExner + du[k,ij])
        end
    end
end

#== velocity tendency du = -q x U - grad B - theta * grad(B) ==#
