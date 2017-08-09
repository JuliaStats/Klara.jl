"""
abstract VariableState{F<:VariateForm}

Root of variable state type hierarchy
"""
abstract type VariableState{F<:VariateForm} end

VariableStateVector{S<:VariableState} = Vector{S}

variate_form{F<:VariateForm}(::Type{VariableState{F}}) = F

"""
Basic univariate variable state type

# Constructors

## BasicUnvVariableState{N<:Number}(value::N)

Construct a basic univariate variable state with some ``value``.

# Examples

```julia
julia> state = BasicUnvVariableState(1.)
Klara.BasicUnvVariableState{Float64}(1.0)

julia> state.value
1.0
```
"""
mutable struct BasicUnvVariableState{N<:Number} <: VariableState{Univariate}
  "Scalar value of basic univariate variable state"
  value::N
end

variate_form(s::Type{BasicUnvVariableState{N}}) where {N<:Number} = Univariate
variate_form(::BasicUnvVariableState) = Univariate

eltype{N<:Number}(::Type{BasicUnvVariableState{N}}) = N
eltype(s::BasicUnvVariableState{N}) where {N<:Number} = N

"""
Basic multivariate variable state type

# Constructors

## BasicMuvVariableState{N<:Number}(value::Vector{N})

Construct a basic multivariate variable state with some ``value``.

## BasicMuvVariableState{N<:Number}(size::Integer, ::Type{N}=Float64)

Construct a basic multivariate variable state with a ``value`` of specified ``size`` and element type.

# Examples

```julia
julia> state = BasicMuvVariableState([1, 2])
Klara.BasicMuvVariableState{Int64}([1,2],2)

julia> state.value
2-element Array{Int64,1}:
 1
 2

julia> state.size
2
```
"""
mutable struct BasicMuvVariableState{N<:Number} <: VariableState{Multivariate}
  "Vector value of basic multivariate variable state"
  value::Vector{N}
  "Integer representing the length of vector value of basic multivariate variable state"
  size::Integer
end

BasicMuvVariableState(value::Vector{N}) where {N<:Number} = BasicMuvVariableState{N}(value, length(value))

BasicMuvVariableState(size::Integer, ::Type{N}=Float64) where {N<:Number} = BasicMuvVariableState{N}(Array{N}(size), size)

variate_form{N<:Number}(::Type{BasicMuvVariableState{N}}) = Multivariate
variate_form(::BasicMuvVariableState) = Multivariate

eltype{N<:Number}(::Type{BasicMuvVariableState{N}}) = N
eltype{N<:Number}(::BasicMuvVariableState{N}) = N

"""
Basic matrix-variate variable state type

# Constructors

## BasicMavVariableState{N<:Number}(value::Matrix{N})

Construct a basic matrix-variate variable state with some ``value``.

## BasicMavVariableState{N<:Number}(size::Tuple, ::Type{N}=Float64)

Construct a basic matrix-variate variable state with a ``value`` of specified ``size`` and element type.

# Examples

```julia
julia> state = BasicMavVariableState(eye(2))
Klara.BasicMavVariableState{Float64}(2x2 Array{Float64,2}:
 1.0  0.0
 0.0  1.0,(2,2))

julia> state.value
2x2 Array{Float64,2}:
 1.0  0.0
 0.0  1.0

julia> state.size
(2,2)
```
"""
mutable struct BasicMavVariableState{N<:Number} <: VariableState{Matrixvariate}
  "Matrix value of basic matrix-variate variable state"
  value::Matrix{N}
  "Tuple containing the dimensions of matrix value of basic matrix-variate variable state"
  size::Tuple{Integer, Integer}
end

BasicMavVariableState(value::Matrix{N}) where {N<:Number} = BasicMavVariableState{N}(value, size(value))

BasicMavVariableState(size::Tuple, ::Type{N}=Float64) where {N<:Number} = BasicMavVariableState{N}(Array{N}(size...), size)

variate_form{N<:Number}(::Type{BasicMavVariableState{N}}) = Matrixvariate
variate_form(::BasicMavVariableState) = Matrixvariate

eltype{N<:Number}(::Type{BasicMavVariableState{N}}) = N
eltype{N<:Number}(::BasicMavVariableState{N}) = N
