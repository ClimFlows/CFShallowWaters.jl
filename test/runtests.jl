import Mooncake, ForwardDiff
import DifferentiationInterface as DI
using NetCDF: ncread

using CFDomains: CFDomains, VoronoiSphere, void
using MutatingOrNot.Allocators

using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
import CFPlanets
import CFTimeSchemes
import ClimFlowsTestCases as CFTestCases

using CFShallowWaters

using JET: @test_call, @test_opt
using Test
using Base: summarysize

macro show_opt_call(expr)
    return esc(quote
        $expr
        @showtime $expr
        @test_opt $expr
        @test_call $expr
        $expr
    end)
end

macro show_call(expr)
    return esc(quote
        $expr
        @showtime $expr
        @test_call $expr
        $expr
    end)
end

cst(args) = map(DI.Constant, args)
prepare(loss, backend, x, args) = DI.prepare_gradient(loss, backend, x, cst(args)...)
gradient(loss, prep, backend, x, args) = DI.gradient(loss, prep, backend, x, cst(args)...)

axpy(a, x::Vector, y) = @. a*x+y
L2(x::Vector) = sum(y->y^2, x)
L2(x::NamedTuple) = mapreduce(L2, +, x)

function setup_RSW(
    sphere, prec ; 
    TestCase = CFTestCases.Williamson91{6},
    Scheme = CFTimeSchemes.RungeKutta4,
    courant = 1.5,
)
    ## physical parameters needed to run the model
    testcase = CFTestCases.testcase(TestCase, prec)
    @info CFTestCases.describe(testcase)
    (; R0, Omega, gH0) = testcase.params

    ## numerical parameters
    @time dx = R0 * CFDomains.laplace_dx(sphere)
    @info "Effective mesh size dx = $(round(dx/1e3)) km"
    dt_dyn = prec(courant * dx / sqrt(gH0))
    @info "Maximum dynamics time step = $(round(dt_dyn)) s"

    ## model setup
    planet = CFPlanets.ShallowTradPlanet(R0, Omega)
    dynamics = CFShallowWaters.RSW(planet, sphere)

    ## initial condition & standard diagnostics
    state0 = dynamics.initialize(CFTestCases.initial_flow, testcase)
    diags = dynamics.diagnostics()
    scheme = Scheme(dynamics)

    return dynamics, diags, state0, scheme, dt_dyn
end

function loss(state, dstate, model, scratch, t)
    CFTimeSchemes.tendencies!(dstate, scratch, model, state, t)
    L2(dstate.ghcov)+L2(dstate.ucov)
end

function loss_FD(t, state, grad, model)
    state0 = map((g,s)->axpy(t,g,s), grad, state)
    scratch = CFTimeSchemes.scratch_space(model, state0, t)
    dstate = CFTimeSchemes.model_dstate(model, state0)
    # @info "loss_FD" typeof(t) typeof(state0) typeof(scratch)
    loss(state0, dstate, model, scratch, t)
end

first_store(smart) = first(values(Allocators.stores(smart)))

meshname, prec = "uni.2deg.mesh.nc", Float64
sphere = VoronoiSphere(DYNAMICO_reader(ncread, DYNAMICO_meshfile(meshname)) ; prec)
@info sphere

dynamics, diags, state0, scheme, dt = setup_RSW(sphere, prec)

@testset "Voronoi adjoint" begin
    t0 = zero(prec)
    dstate = CFTimeSchemes.model_dstate(dynamics, state0)
    scratch = CFTimeSchemes.scratch_space(dynamics, state0, t0)
    
    @show_opt_call CFTimeSchemes.tendencies!(dstate, scratch, dynamics, state0, t0)
    @show_opt_call loss(state0, dstate, dynamics, scratch, t0)

    smart = SmartAllocator()
    loss(state0, dstate, dynamics, smart, t0)
#    Allocators.debug_store(:test, first_store(smart))
    @show_call loss(state0, dstate, dynamics, smart, t0)
#    Allocators.debug_store(:test, first_store(smart))

    backend = DI.AutoMooncake(; config=nothing)

    args = dstate, dynamics, smart, t0
    prep = prepare(loss, backend, state0, args)
    grad = @show_call gradient(loss, prep, backend, state0, args)
    @info "" summarysize(prep) summarysize(smart) summarysize(args)

    args = dstate, dynamics, scratch, t0
    prep = prepare(loss, backend, state0, args)
    grad = @show_call gradient(loss, prep, backend, state0, args)
    @info "" summarysize(prep) summarysize(scratch) summarysize(args)

    FD = DI.AutoForwardDiff()
    args2 = state0, grad, dynamics
    prep2 = prepare(loss_FD, FD, t0, args2);
    L2_grad_ = @show_call gradient(loss_FD, prep2, FD, t0, args2);
    
    @info "Voronoi adjoint" L2(grad) L2_grad_
    @test L2(grad) ≈ L2_grad_
end
