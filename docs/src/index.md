# Index

NDInterpolations.jl is a library for interpolating arbitrarily high dimensional array data. The domain of this interpolation is a (hyper)rectangle. Support is included for efficient evaluation at multiple points in the domain through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

For one dimensional interpolation see also [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl).

## API

An `NDInterpolation` is defined by a tuple of interpolation dimensions and the data `u` to interpolate.

```julia
using NDInterpolations

t1 = cumsum(rand(5))
t2 = cumsum(rand(7))

interpolation_dimensions = (
    LinearInterpolationDimension(t1),
    LinearInterpolationDimension(t2)
)

# The outputs will be vectors of length 2
u = rand(5, 7, 2)

interp = NDInterpolation(u, interpolation_dimensions)
```

Evaluation of this vector valued interpolation can be done in place or out of place.

```julia
interp(0.5, 0.5)

out = zeros(2)
interp(out, 0.5, 0.5)
```

If we provide `t_eval` for the interpolation dimensions, we can evaluate at these points either 'zipped' (where all `t_eval` must be of the same length) or as a grid defined by the Cartesian product of the `t_eval`.

```julia
interpolation_dimensions = (
    LinearInterpolationDimension(t1; t_eval = range(first(t1), last(t1); length = 100)),
    LinearInterpolationDimension(t2; t_eval = range(first(t2), last(t2); length = 100))
)

interp = NDInterpolation(u, interpolation_dimensions)

# Out of place zipped evaluation
eval_unstructured(interp) # Yields Matrix of size (100, 2)

# In place grid evaluation
out = zeros(100, 100, 2)
eval_grid!(out, interp)
```

## Available interpolations

The interpolation types are given by the corresponding interpolation dimension type.

  - `LinearInterpolationDimension(t)`: Linear interpolation in the sense of bilinear, trilinear interpolation etc.
  - `ConstantInterpolationDimension(t)`: An interpolation with a constant value in each interval between `t` points. The Boolean option `left` (default `true`) can be used to indicate which side of the interval in which the input lies determines the output value.
  - `BSplineInterpolationDimension(t, degree)`: Interpolation using BSpline basis functions. The input values `t` are interpreted as knots, and optionally knot multiplicities can be supplied. Per dimension a degree can be specified. Note that for an `NDInterpolation` of this type, the size of `u` for a certain dimension is equal to `sum(multiplicities) - degree - 1`. This interpolation dimension type can also be used to define NURBS, by passing `global_cache = NURBSWeights(weights)` to the `NDInterpolation` constructor.
