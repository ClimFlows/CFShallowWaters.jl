using MutatingOrNot: void
using SHTnsSpheres: SHTnsSphere
using CFDomains: VoronoiSphere
using ClimFlowsData: DYNAMICO_reader
using CFTimeSchemes: tendencies!
import ClimFlowsTestCases as CFTestCases
using CFPlanets

using CFShallowWaters
using Test
using NetCDF: ncread

choices = (
    precision = Float32,
    meshname = "uni.1deg.mesh.nc",
    TestCase = CFTestCases.Williamson91{6},
)

testcase = CFTestCases.testcase(choices.TestCase, Float32)
(; R0, Omega) = testcase.params
planet = CFPlanets.ShallowTradPlanet(R0, Omega)

reader = DYNAMICO_reader(ncread, choices.meshname)
sphere = VoronoiSphere(reader; prec = choices.precision)
@info sphere

model = CFShallowWaters.RSW(planet, sphere)
@info model

ghcov, ucov = similar(sphere.Ai), similar(sphere.le_de)
state = (; ghcov, ucov)

dstate = tendencies!(void, model, state, void, nothing)
dstate = tendencies!(dstate, model, state, void, nothing)

@testset "CFShallowWaters.jl" begin
    # Write your tests here.
end

