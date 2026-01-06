"""
    initial_state = initialize(model, fun, args...)
    initial_state = model.initialize(fun, args...)

Returns an discretized, model representation of the initial state described by function `fun`. `fun` is expected to follow the pattern:
    gh, ulon, ulat = fun(lon, lat, args...)
with `gh` geopotential, `ulon` zonal velocity and `ulat` meridional velocity.

Details of the representation depend on the domain and numerical method. Since the domain is devoid of metric information (e.g. a unit sphere),
metric coefficients are factored into the model representation, which is thus not directly interpretable in terms of observable quantities (e.g. wind).
It may consist of several arrays, consisting of spherical harmonic coefficients, a discrete exterior calculus representation...

Observable quantities may be computed using `model.diagnostics()`
"""
initialize(model, fun, args...) = initialize_SW(model.domain, model, fun, args...)

function initialize_SW(domain::VoronoiSphere, model, fun, args...)
    gh, ulon, ulat = allocate_fields( (:scalar, :vector, :vector), domain, eltype(domain) )
    for ij in eachindex(gh)
        gh[ij], ulon_, ulat_ = fun(domain.lon_i[ij], domain.lat_i[ij], args...)
    end
    for ij in eachindex(ulon)
        gh_, ulon[ij], ulat[ij] = fun(domain.lon_e[ij], domain.lat_e[ij], args...)
    end
    return observable_to_prognostic(model.planet, domain, (; gh, ulon, ulat))
end

function observable_to_prognostic(planet, domain::VoronoiSphere, (; gh, ulon, ulat))
    (; de, Ai) = domain
    ghcov, ucov = copy(gh), allocate_field(:vector, domain, eltype(ulon))
    for ij in eachindex(ghcov)
        a = scale_factor(planet, domain.lon_i[ij], domain.lat_i[ij])
        ghcov[ij] = a*a*gh[ij]*Ai[ij]
    end
    for ij in eachindex(ucov)
        a = scale_factor(planet, domain.lon_e[ij], domain.lat_e[ij])
        angle = domain.angle_e[ij]
        ucov[ij] = (cos(angle)*ulon[ij] + sin(angle)*ulat[ij])*a*de[ij]
    end
    return (; ghcov, ucov)
end
