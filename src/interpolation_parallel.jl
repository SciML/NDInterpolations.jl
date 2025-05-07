"""
    function eval_unstructured(
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the unstructured set of points defined by `t_eval`
in the interpolation dimensions out of place. That is, `t_eval` must have the same
length for each interpolation dimension and the interpolation is evaluated at the `zip` if these `t_eval`.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_unstructured(interp::NDInterpolation; kwargs...)
    n_points = length(first(interp.interp_dims).t_eval)
    out = similar(interp.u, (n_points, get_output_size(interp)...))
    eval_unstructured!(out, interp; kwargs...)
end

"""
    function eval_unstructured!(
        out::AbstractArray,
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the unstructured set of points defined by `t_eval`
in the interpolation dimensions in place. That is, `t_eval` must have the same
length for each interpolation dimension and the interpolation is evaluated at the `zip` if these `t_eval`.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_unstructured!(
        out::AbstractArray,
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)
) where {N_in}
    validate_derivative_orders(derivative_orders, interp; multi_point = true)
    backend = get_backend(out)
    @assert all(i -> length(interp.interp_dims[i].t_eval) == size(out, 1), N_in) "The t_eval of all interpolation dimensions must have the same length as the first dimension of out."
    @assert size(out)[2:end]==get_output_size(interp) "The size of the last N_out dimensions of out must be the same as the output size of the interpolation."
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
        false,
        ndrange = size(out, 1)
    )
    synchronize(backend)
    return out
end

"""
    function eval_grid(
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the Cartesian product of the `t_eval` of the interpolation dimensions
out of place.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_grid(interp::NDInterpolation{N_in}; kwargs...) where {N_in}
    grid_size = map(itp_dim -> length(itp_dim.t_eval), interp.interp_dims)
    out = similar(interp.u, (grid_size..., get_output_size(interp)...))
    eval_grid!(out, interp; kwargs...)
end

"""
    function eval_grid!(
        out::AbstractArray,
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the Cartesian product of the `t_eval` of the interpolation dimensions
in place.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_grid!(
        out::AbstractArray,
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)
) where {N_in}
    validate_derivative_orders(derivative_orders, interp; multi_point = true)
    backend = get_backend(out)
    @assert all(i -> size(out, i) == length(interp.interp_dims[i].t_eval), N_in) "For the first N_in dimensions of out the length must match the t_eval of the corresponding interpolation dimension."
    @assert size(out)[(N_in + 1):end]==get_output_size(interp) "The size of the last N_out dimensions of out must be the same as the output size of the interpolation."
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
        true,
        ndrange = size(out)[1:N_in]
    )
    synchronize(backend)
    return out
end

@kernel function eval_kernel(
        out,
        @Const(A),
        derivative_orders,
        eval_grid
)
    N_in = length(A.interp_dims)
    N_out = ndims(A.u) - N_in

    k = @index(Global, NTuple)

    if eval_grid
        t_eval = ntuple(i -> A.interp_dims[i].t_eval[k[i]], N_in)
        idx_eval = ntuple(i -> A.interp_dims[i].idx_eval[k[i]], N_in)
    else
        t_eval = ntuple(i -> A.interp_dims[i].t_eval[only(k)], N_in)
        idx_eval = ntuple(i -> A.interp_dims[i].idx_eval[only(k)], N_in)
    end

    if iszero(N_out)
        out[k...] = _interpolate!(
            make_out(A, t_eval), A, t_eval, idx_eval, derivative_orders, k)
    else
        _interpolate!(
            view(out, k..., ..),
            A, t_eval, idx_eval, derivative_orders, k)
    end
end
