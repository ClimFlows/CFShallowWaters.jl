#======== All =========#

scratch_space(model, state) = scratch_SW(model.domain, state)
tendencies!(dstate, model, state, scratch, t) = tendencies_SW!(dstate, state, scratch, model, model.domain)

#========= Voronoi ========#

function same(x,y)
    @assert x==y
    return x
end

function scratch_SW(domain::VoronoiSphere, (; ucov, ghcov))
    F = same(eltype(ucov), eltype(ghcov))
    allocate_fields((qv=:dual, qe=:vector, U=:vector, B=:scalar, gh=:scalar), domain, F)
end

function tendencies_SW!( dstate, (; ucov,ghcov), scratch, model, mesh::VoronoiSphere)
    hodges = mesh.le_de
    inv_Ai = mesh.inv_Ai
    radius = model.planet.radius

    gh = scratch.gh
    @. gh = ghcov
    @. gh = inv_Ai*gh

    U = massflux!(scratch.U, ucov, gh, radius, mesh.edge_left_right, hodges)
    B = bernoulli!(scratch.B, gh, ucov, radius, mesh.primal_deg, mesh.Ai, hodges, mesh.primal_edge)
    qv = voronoi_potential_vorticity!(scratch.qv, model.fcov, ucov, gh, mesh.Av, mesh.dual_vertex, mesh.dual_edge, mesh.dual_ne, mesh.Riv2)
    ducov = voronoi_du!(dstate.ucov, scratch.qe, qv, U, B, mesh.edge_down_up,
        mesh.edge_left_right, mesh.trisk_deg, mesh.trisk, mesh.wee)
    dghcov = voronoi_dm!(dstate.ghcov, U, mesh.Ai, mesh.primal_deg, mesh.primal_edge, mesh.primal_ne)

#    @. dghcov = inv_Ai * dghcov

    return (ghcov=dghcov, ucov=ducov)
end

#=
function tendencies_SW( (; ucov, ghcov), model, mesh::VoronoiSphere)
    hodges = mesh.le_de
    radius = model.planet.radius
    U = massflux(ucov, ghcov, radius, mesh.edge_left_right, hodges)
    B = bernoulli(ghcov, ucov, radius, mesh.primal_deg, mesh.Ai, hodges, mesh.primal_edge)
    qv = voronoi_potential_vorticity!(void, model.fcov, ucov, ghcov, mesh.Av, mesh.dual_vertex, mesh.dual_edge, mesh.dual_ne, mesh.Riv2)
    ducov = voronoi_du!(void, qv, U, B, mesh.edge_down_up,
        mesh.edge_left_right, mesh.trisk_deg, mesh.trisk, mesh.wee)
    dghcov = voronoi_dm!(void, U, mesh.Ai, mesh.primal_deg, mesh.primal_edge, mesh.primal_ne)
    return (ghcov=dghcov, ucov=ducov)
end
=#

#== Voronoi, mass flux ==#

massflux!(::Void, ucov, m, radius, left_right, hodges) =
    massflux!(similar(ucov), ucov, m, radius, left_right, hodges)

function massflux!(U::AbstractVector, ucov, m, radius, left_right, hodges)
    @fast for ij in eachindex(hodges)
        left, right = left_right[1,ij], left_right[2,ij]
        U[ij] = inv(2*radius*radius)*(m[left]+m[right])*ucov[ij]*hodges[ij]
    end
    return U
end

massflux!(backend, U::AbstractMatrix, args...) = massflux_3D!(backend, U, args...)

@loops function massflux_3D!(_, U, ucov, m, radius, left_right, hodges)
    let range = axes(U,2)
        @fast for ij in range
            nz = size(U,1)
            left, right = left_right[1,ij], left_right[2,ij]
            for k=1:nz
                U[k,ij] = inv(2*radius*radius)*(m[k,left]+m[k,right])*ucov[k,ij]*hodges[ij]
            end
        end
    end
end

#== Voronoi, kinetic energy ==#

bernoulli!(::Void, gh, ucov, radius, degree, areas, hodges, edges) =
    bernoulli!(similar(gh), gh, ucov, radius, degree, areas, hodges, edges)

function bernoulli!(B::AbstractVector, gh, ucov, radius, degree, areas, hodges, edges)
    inv_r2 = inv(radius*radius)
    @fast for ij in eachindex(areas)
        deg, inv_area = degree[ij], inv(4*areas[ij])
        @unroll deg in 5:7 B[ij] = inv_r2 * (
            gh[ij] + inv_area*sum(hodges[edges[e,ij]]*ucov[edges[e,ij]]^2 for e=1:deg) )
    end
    return B
end

bernoulli!(backend, B::AbstractMatrix, args...) = bernoulli_3D!(backend, B, args...)

@loops function bernoulli_3D!(_, B, ucov, radius, degree, areas, hodges, edges)
    let range = axes(B,2)
        nz = size(B,1)
        @fast for ij in range
            deg, aa = degree[ij], inv(4*areas[ij]*radius*radius)
            @unroll deg in 5:7 begin
                ee = (edges[e,ij] for e=1:deg)
                hh = (hodges[ee[e]] for e=1:deg)
                @simd for k=1:nz
                    B[k,ij] += aa * sum(hh[e]*ucov[k,ee[e]]^2 for e=1:deg)
                end
            end
        end
    end
end

#== Voronoi, Poisson bracket ==#

#== potential vorticity q = curl(ucov)/m ==#

voronoi_potential_vorticity!(::Void, fv, ucov, m, areas, cells, edges, signs, weights) =
    voronoi_potential_vorticity!(similar(fv, eltype(ucov)), fv, ucov, m, areas, cells, edges, signs, weights)

function voronoi_potential_vorticity!(qv::AbstractVector, fv, ucov, m, areas, cells, edges, signs, weights)
    @fast @unroll for ij in eachindex(qv)
        zeta = sum(ucov[edges[edge,ij]]*signs[edge,ij] for edge=1:3 )
        mv =  areas[ij]*sum(m[cells[vertex,ij]]*weights[vertex,ij] for vertex=1:3 )
        qv[ij] = inv(mv)*(fv[ij]+zeta)
    end
    return qv
end

@loops function voronoi_potential_vorticity!(_, qv::AbstractMatrix, fv, ucov, m, areas, cells, edges, signs, weights)
    let (krange, ijrange) = axes(qv)
        @fast @unroll for ij in ijrange
            F, nz = eltype(qv), size(qv,1)
            ee = ( edges[edge,ij] for edge=1:3 )
            ss = ( F(signs[edge,ij]) for edge=1:3 )
            cc = ( cells[vertex,ij] for vertex=1:3 )
            ww = ( weights[vertex,ij] for vertex=1:3 )
            @simd for k in krange
                zeta = sum(ucov[k,ee[edge]]*ss[edge] for edge=1:3 )
                mv =  areas[ij]*sum(m[k,cc[vertex]]*ww[vertex] for vertex=1:3 )
                qv[k,ij] = inv(mv)*(fv[ij]+zeta)
            end
        end
    end
end

#== mass tendency dm = -div(U) ==#

voronoi_dm!(::Void, U, areas, degree, edges, signs) =
    voronoi_dm!(similar(areas, eltype(U)), U, areas, degree, edges, signs)

function voronoi_dm!(dm::AbstractVector, U, areas, degree, edges, signs)
    @fast for ij in eachindex(dm)
        deg = degree[ij]
#        @unroll deg in 5:7 dm[ij] = -inv(areas[ij]) * sum( signs[e,ij]*U[edges[e,ij]] for e=1:deg )
        @unroll deg in 5:7 dm[ij] = -sum( signs[e,ij]*U[edges[e,ij]] for e=1:deg )
    end
    return dm
end

@loops function voronoi_dm_3D!(_, dm, U, areas, degree, edges, signs)
    let (krange, ijrange) = (axes(dmass,1), axes(dmass,2))
        F = eltype(dm)
        @fast for ij in ijrange
            deg, aa = degree[ij], inv(areas[ij])
            @unroll deg in 5:7 begin
                ee = ( edges[e,ij] for e=1:deg )
                ss = ( F(signs[e,ij]) for e=1:deg )
                @simd for k in krange
                    dm[k,ij] = -aa * sum( ss[e]*U[k,ee[e]] for e=1:deg )
                end
            end
        end
    end
end

@loops function voronoi_dm_3D!(_, dmass, U, B, areas, degree, edges, signs, left_right)
    let (krange, ijrange) = (axes(dmass,1), axes(dmass,2))
        @views begin
            dm     = dmass[:,:,1]
            dTheta = dmass[:,:,2]
            theta  = B[:,:,2]
            left   = left_right[1,:]
            right  = left_right[2,:]
        end

        F = eltype(dm)
        @fast for ij in ijrange
            deg, aa = degree[ij], inv(areas[ij])
            aa2 = half(aa)
            @unroll deg in 5:7 begin
                ee = ( edges[e,ij] for e=1:deg )
                ss = ( F(signs[e,ij]) for e=1:deg )
                ll = ( left[ee[e]] for e=1:deg )
                rr = ( right[ee[e]] for e=1:deg )
                @simd for k in krange
                    dm[k,ij] = -aa * sum( ss[e]*U[k,ee[e]] for e=1:deg )
                    dTheta[k,ij] = -aa2 * sum( (ss[e]*U[k,ee[e]])*(theta[k,ll[e]]+theta[k,rr[e]]) for e=1:deg )
                end
            end
        end

    end
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
