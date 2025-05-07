module NDInterpolations
using KernelAbstractions # Keep as dependency or make extension?
using Adapt: @adapt_structure
using EllipsisNotation
using RecipesBase

abstract type AbstractInterpolationDimension end
abstract type AbstractGlobalCache end

struct TrivialGlobalCache <: AbstractGlobalCache end

"""
    NDInterpolation(interp_dims, u)

The interpolation object containing the interpolation dimensions and the data to interpolate `u`.
Given the number of interpolation dimensions `N_in`, for first `N_in` dimensions of `u`
the size of `u` along that dimension must match the length of `t` of the corresponding interpolation dimension.

## Arguments

  - `interp_dims`: A tuple of identically typed interpolation dimensions.
  - `u`: The array to be interpolated.
"""
struct NDInterpolation{
    N_in, N_out,
    ID <: AbstractInterpolationDimension,
    gType <: AbstractGlobalCache,
    uType <: AbstractArray
}
    u::uType
    interp_dims::NTuple{N_in, ID}
    global_cache::gType
    function NDInterpolation(u, interp_dims, global_cache)
        if interp_dims isa AbstractInterpolationDimension
            interp_dims = (interp_dims,)
        end
        N_in = length(interp_dims)
        N_out = ndims(u) - N_in
        @assert N_out≥0 "The number of dimensions of u must be at least the number of interpolation dimensions."
        validate_size_u(interp_dims, u)
        validate_global_cache(global_cache, interp_dims, u)
        new{N_in, N_out, eltype(interp_dims), typeof(global_cache), typeof(u)}(
            u, interp_dims, global_cache
        )
    end
end

# Constructor with optional global cache
function NDInterpolation(u, interp_dims; global_cache = TrivialGlobalCache())
    NDInterpolation(u, interp_dims, global_cache)
end

@adapt_structure NDInterpolation

include("interpolation_dimensions.jl")
include("spline_utils.jl")
include("interpolation_utils.jl")
include("interpolation_methods.jl")
include("interpolation_parallel.jl")
include("plot_rec.jl")

# Multiple `t` arguments to tuple (can these 2 be done in 1?)
function (interp::NDInterpolation)(t_args::Vararg{Number}; kwargs...)
    interp(t_args; kwargs...)
end

function (interp::NDInterpolation)(
        out::AbstractArray, t_args::Vararg{Number}; kwargs...)
    interp(out, t_args; kwargs...)
end

# In place single input evaluation
function (interp::NDInterpolation{N_in})(
        out::Union{Number, AbstractArray{<:Number}},
        t::Tuple{Vararg{Number, N_in}};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)
) where {N_in}
    validate_derivative_orders(derivative_orders, interp)
    idx = get_idx(interp.interp_dims, t)
    @assert size(out)==size(interp.u)[(N_in + 1):end] "The size of out must match the size of the last N_out dimensions of u."
    _interpolate!(out, interp, t, idx, derivative_orders, nothing)
end

# Out of place single input evaluation
function (interp::NDInterpolation)(t::Tuple{Vararg{Number}}; kwargs...)
    out = make_out(interp, t)
    interp(out, t; kwargs...)
end

export NDInterpolation, LinearInterpolationDimension, ConstantInterpolationDimension,
       BSplineInterpolationDimension, NURBSWeights,
       eval_unstructured, eval_unstructured!, eval_grid, eval_grid!

end # module NDInterpolations
