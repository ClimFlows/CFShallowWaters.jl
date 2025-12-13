module CFShallowWaters

using ManagedLoops: @unroll
using CFPlanets: ShallowTradPlanet, coriolis, scale_factor
using CFDomains: VoronoiSphere, allocate_fields, allocate_field

using CookBooks
import MemberFunctions: WithMembers, member_functions

macro fast(code)
    debug = haskey(ENV, "GF_DEBUG") && (ENV["GF_DEBUG"] != "")
    return debug ? esc(code) : esc(quote
        @inbounds $code
    end)
end

abstract type AbstractSW <: WithMembers end
@inline member_functions(::Type{<:AbstractSW}) = (; scratch_space, initialize, diagnostics)

struct Traditional_RSW{Planet,Domain,Fcov} <: AbstractSW
    planet::Planet
    domain::Domain
    fcov::Fcov # coriolis factor multiplied by Jacobian

    function Traditional_RSW(planet::P, domain::D) where {P,D}
        fcov = coriolis_cov(eltype(domain), planet, domain)
        new{P,D,typeof(fcov)}(planet, domain, fcov)
    end
end

Base.show(io::IO, ::Type{<:Traditional_RSW{Planet, Domain}}) where {Planet, Domain} = print(io,
    "Traditional_RSW{$Planet, $Domain}")

Base.show(io::IO, trad::Traditional_RSW{Planet}) where {Planet} = print(io,
    "Traditional_RSW(planet=$(trad.planet), domain=$(trad.domain)")

struct Oblate_RSW{Planet,Domain,RLambda,RPhi} <: AbstractSW
    planet::Planet
    domain::Domain
    fcov::Matrix{Float64} # coriolis factor multiplied by Jacobian
    Rlambda::RLambda      # zonal metric factor
    Rphi::RPhi            # meridional metric factor
end

RSW(planet::ShallowTradPlanet, domain) = Traditional_RSW(planet, domain)

coriolis_cov(F, planet, domain::VoronoiSphere) =
    coriolis_voronoi(F, planet, domain.lon_v, domain.lat_v, domain.Av)
coriolis_voronoi(F, planet, lon_v, lat_v, Av) = F.(coriolis(planet, lon_v, lat_v) .* Av)

scratch_space(model, state) = scratch_SW(model.domain, state)
tendencies!(dstate, model, state, scratch, t) = tendencies_SW!(dstate, state, scratch, model, model.domain)

# implemented in Voronoi submodule
function scratch_SW end
function tendencies_SW! end

include("julia/initialize.jl")
include("julia/voronoi.jl")
include("julia/diagnostics.jl")

end # module

#=

Models.allocate_state(model::AbstractSW) =
    allocate_SW_state(model.domain, eltype(model.domain))
Models.backend(::AbstractSW) = GFLoops.default_backend()

allocate_SW_state(domain::SpectralDomain) = allocate_fields((:cvector, :scalar), domain)
allocate_SW_scratch(domain::SpectralDomain) =
    allocate_fields((:cvector, :scalar, :scalar), domain)

allocate_SW_state(domain::VoronoiSphere, F::Type{<:Real}) =
    allocate_fields((:vector, :scalar), domain, F)

coriolis_cov(F, planet, domain::SpectralDomain) = F.(coriolis(planet, domain.lat))
coriolis_voronoi(F, planet, lat_v::W, Av::W) where {W<:WrappedArray} =
    wrap_array(coriolis_voronoi(F, planet, lat_v.data, Av.data), lat_v)
    
=#
