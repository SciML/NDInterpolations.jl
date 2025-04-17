using NDInterpolations: AbstractInterpolationDimension
using NDInterpolations
using Random

function test_globally_constant(
        ID::Type{<:AbstractInterpolationDimension}; args1 = [], args2 = [], kwargs1 = [], kwargs2 = [])
    t1 = [-3.14, 1.0, 3.0, 7.6, 12.8]
    t2 = [-2.71, 1.41, 12.76, 50.2, 120.0]

    u = if ID == BSplineInterpolationDimension
        fill(2.0, 4 + args1[1], 4 + args2[1])
    else
        fill(2.0, 5, 5)
    end

    # Evaluation in data points
    itp_dims = (
        ID(t1, args1...; t_eval = t1, kwargs1...),
        ID(t2, args2...; t_eval = t2, kwargs2...)
    )

    itp = NDInterpolation(u, itp_dims)
    @test all(x -> isapprox(x, 2.0; atol = 1e-10), eval_grid(itp))
    @test all(
        x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (1, 0)))
    @test all(
        x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (0, 1)))

    # Evaluation between data points
    itp_dims = (
        ID(t1, args1...; t_eval = t1[1:(end - 1)] + diff(t1) / 2, kwargs1...),
        ID(t2, args2...; t_eval = t2[1:(end - 1)] + diff(t2) / 2, kwargs2...)
    )
    itp = NDInterpolation(u, itp_dims)
    @test all(x -> isapprox(x, 2.0; atol = 1e-10), eval_grid(itp))
    @test all(
        x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (1, 0)))
    @test all(
        x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (0, 1)))
end

function test_analytic(itp::NDInterpolation{N_in}, f) where {N_in}
    # Evaluation in data points
    ts = ntuple(dim_in -> itp.interp_dims[dim_in].t, N_in)
    for t in Iterators.product(ts...)
        @test itp(t) ≈ f(t...)
    end

    # Evaluation between data points
    ts_ = ntuple(dim_in -> ts[dim_in][1:(end - 1)] + diff(ts[dim_in]) / 2, N_in)
    for t in Iterators.product(ts_...)
        @test itp(t) ≈ f(t...)
    end
end

@testset "Linear Interpolation" begin
    test_globally_constant(LinearInterpolationDimension)

    f(t1, t2) = 3.0 + 2.3t1 - 4.7t2

    Random.seed!(1)
    t1 = cumsum(rand(10))
    t2 = cumsum(rand(10))

    itp_dims = (
        LinearInterpolationDimension(t1),
        LinearInterpolationDimension(t2)
    )
    u = f.(t1, t2')
    itp = NDInterpolation(u, itp_dims)
    test_analytic(itp, f)
end

@testset "BSpline Interpolation" begin
    test_globally_constant(
        BSplineInterpolationDimension, args1 = [2], args2 = [3],
        kwargs1 = [:max_derivative_order_eval => 1],
        kwargs2 = [:max_derivative_order_eval => 1]
    )

    f(t1, t2, t3) = t1^2 + t2^2 + t3^2

    u = zeros(3, 3, 3)
    u[2, 2, 2] = -3
    for I in Iterators.product((1, 3), (1, 3), (1, 3))
        u[I...] = 3
    end

    itp_dim = BSplineInterpolationDimension([-1.0, 1.0], 2)
    itp = NDInterpolation(u, (itp_dim, itp_dim, itp_dim))
    test_analytic(itp, f)
end
