@kernel function expand_knot_vector_kernel(
        knots_all,
        @Const(knot_values),
        @Const(multiplicities)
)
    i = @index(Global, Linear)
    knot_value = knot_values[i]

    mult_sum = 0
    idx_end = i - 1
    for j in 1:idx_end
        mult_sum += multiplicities[j]
    end

    idx_start = mult_sum + 1
    idx_end = mult_sum + multiplicities[i]
    for k in idx_start:idx_end
        knots_all[k] = knot_value
    end
end

safe_div(a, b) = iszero(b) ? zero(promote_type(typeof(a), typeof(b))) : a / b

function cox_de_boor(
        basis_function_values, knots_all, t, idx, degree, d, k, deriv
)
    T = eltype(basis_function_values)
    if (k ≤ degree - d) || (k == degree + 2)
        zero(T)
    else
        i = idx + k - degree - 1
        tᵢ = knots_all[i]
        tᵢ₊₁ = knots_all[i + 1]
        tᵢ₊ₚ = knots_all[i + d]
        tᵢ₊ₚ₊₁ = knots_all[i + d + 1]
        if deriv
            T(d * (safe_div(basis_function_values[k], tᵢ₊ₚ - tᵢ) -
               safe_div(basis_function_values[k + 1], tᵢ₊ₚ₊₁ - tᵢ₊₁)))
        else
            T(basis_function_values[k] * safe_div(t - tᵢ, tᵢ₊ₚ - tᵢ) +
              basis_function_values[k + 1] * safe_div(tᵢ₊ₚ₊₁ - t, tᵢ₊ₚ₊₁ - tᵢ₊₁))
        end
    end
end

# Basis function `Bᵢₚ` (Where `p` is the degree) has support on the interval `[tᵢ, tᵢ₊ₚ₊₁)`
# where these `tᵢ` come from `itp_dim.knots_all`. That means that an input `t ∈ [tⱼ, tⱼ₊₁)` (where `j = idx`)
# lies in the support of `Bᵢₚ` for `i = j - p, …, j`, i.e. `p + 1` of the basis functions.
#
# The basis functions are computed recursively using the Cox - de Boor formula in tuples of
# length `p + 2` as follows:
# (0, …, 0, Bⱼ₀, 0) = (0, … 0, 1, 0)
# (0, …, 0, Bⱼ₋₁ ₁, Bⱼ₁, 0)
#          ⋮
# (Bⱼ₋ₚ ₚ, Bⱼ₋ₚ₊₁ ₚ, …, Bⱼₚ, 0)
# 
# The trailing zero is just a convenience for the algorithm and is removed in the output
function get_basis_function_values(
        itp_dim::BSplineInterpolationDimension,
        t::Number,
        idx::Integer,
        derivative_order::Integer,
        multi_point_index::Nothing,
        dim_in::Integer
)
    (; degree, knots_all) = itp_dim
    T = promote_type(typeof(t), eltype(itp_dim.basis_function_eval))
    degree_plus_1 = degree + 1

    if derivative_order > degree
        return ntuple(_ -> zero(T), degree_plus_1)
    end

    degree_plus_2 = degree + 2

    # Degree 0 basis function values
    basis_function_values = ntuple(
        k -> (k == degree_plus_1) ? one(T) : zero(T),
        degree_plus_2
    )

    # Higher order basis function values
    for d in 1:degree
        deriv = d > degree - derivative_order
        basis_function_values = ntuple(
            k -> cox_de_boor(
                basis_function_values, knots_all, t, idx, degree, d, k, deriv
            ),
            degree_plus_2
        )
    end

    basis_function_values[1:degree_plus_1]
end

# Get the basis function values for one point in an
# unstructured multi point evaluation (given by the scalar multi point index)
function get_basis_function_values(
        itp_dim::BSplineInterpolationDimension,
        t::Number,
        idx::Integer,
        derivative_order::Integer,
        multi_point_index::Number,
        dim_in::Integer
)
    view(itp_dim.basis_function_eval,
        multi_point_index, :, derivative_order + 1)
end

# Get the basis function values for one point in a
# grid evaluation (given by the tuple multi point index)
function get_basis_function_values(
        itp_dim::BSplineInterpolationDimension,
        t::Number,
        idx::Integer,
        derivative_order::Integer,
        multi_point_index::NTuple{N_in, <:Integer},
        dim_in::Integer
) where {N_in}
    view(itp_dim.basis_function_eval,
        multi_point_index[dim_in], :, derivative_order + 1)
end

# Get all basis function values to evaluate a BSpline interpolation in t
function get_basis_function_values_all(
        A::NDInterpolation{N_in, N_out, <:BSplineInterpolationDimension},
        t::Tuple{Vararg{Number, N_in}},
        idx::NTuple{N_in, <:Integer},
        derivative_orders::NTuple{N_in, <:Integer},
        multi_point_index
) where {N_in, N_out}
    ntuple(
        dim_in -> get_basis_function_values(
            A.interp_dims[dim_in], t[dim_in], idx[dim_in], derivative_orders[dim_in], multi_point_index, dim_in
        ),
        N_in
    )
end

function set_basis_function_eval!(itp_dim::BSplineInterpolationDimension)::Nothing
    backend = get_backend(itp_dim.t_eval)
    basis_function_eval_kernel(backend)(
        itp_dim,
        ndrange = (length(itp_dim.t_eval), itp_dim.max_derivative_order_eval + 1)
    )
    synchronize(backend)
    return nothing
end

@kernel function basis_function_eval_kernel(
        itp_dim
)
    i, derivative_order_plus_1 = @index(Global, NTuple)

    itp_dim.basis_function_eval[i,
    :,
    derivative_order_plus_1] .= get_basis_function_values(
        itp_dim,
        itp_dim.t_eval[i],
        itp_dim.idx_eval[i],
        derivative_order_plus_1 - 1,
        nothing,
        0
    )
end

# Wrapper for a BSplineInterpolationDimension which acts as a vector of the values
# of the i-th basis function in the `itp_dim.t_eval`. 
struct BasisFunctionVector{ID <: BSplineInterpolationDimension, T} <: AbstractVector{T}
    itp_dim::ID
    i::Int
    derivative_order_eval::Int
    function BasisFunctionVector(itp_dim, i, derivative_order_eval)
        n_basis_functions = get_n_basis_functions(itp_dim)
        @assert 1≤i≤n_basis_functions "The itp_dim has only $n_basis_functions basis functions, got $i."
        @assert 0≤derivative_order_eval≤itp_dim.max_derivative_order_eval "The itp_dim has max_derivative_order_eval = $(itp_dim.max_derivative_order_eval), got $derivative_order_eval."
        new{typeof(itp_dim), eltype(itp_dim.basis_function_eval)}(
            itp_dim, i, derivative_order_eval
        )
    end
end

Base.length(bfv::BasisFunctionVector) = length(bfv.itp_dim.t_eval)
Base.size(bfv::BasisFunctionVector) = (length(bfv),)

function Base.getindex(bfv::BasisFunctionVector, j)
    (; itp_dim, i, derivative_order_eval) = bfv
    (; basis_function_eval, idx_eval, degree) = itp_dim
    idx = idx_eval[j]
    if i ≤ idx ≤ i + degree
        basis_function_eval[j, degree + i - idx + 1, derivative_order_eval + 1]
    else
        zero(eltype(basis_function_eval))
    end
end

function get_n_basis_functions(itp_dim::BSplineInterpolationDimension)
    length(itp_dim.knots_all) - itp_dim.degree - 1
end

"""
    NURBSWeights(weights::AbstractArray)

Weights associated with the control points to define a NURBS geometry.
"""
struct NURBSWeights{W <: AbstractArray} <: AbstractGlobalCache
    weights::W
end
