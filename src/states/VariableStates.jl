### Abstract variable states

abstract VariableState{F<:VariateForm}

variate_form{F<:VariateForm}(::Type{VariableState{F}}) = F

### Basic variable state subtypes

## BasicUnvVariableState

type BasicUnvVariableState{N<:Number} <: VariableState{Univariate}
  value::N
end

variate_form{N<:Number}(s::Type{BasicUnvVariableState{N}}) = Univariate
variate_form{N<:Number}(s::BasicUnvVariableState{N}) = Univariate

Base.eltype{N<:Number}(::Type{BasicUnvVariableState{N}}) = N
Base.eltype{N<:Number}(s::BasicUnvVariableState{N}) = N

## BasicMuvVariableState

type BasicMuvVariableState{N<:Number} <: VariableState{Multivariate}
  value::Vector{N}
  size::Int
end

BasicMuvVariableState{N<:Number}(value::Vector{N}) = BasicMuvVariableState{N}(value, length(value))

BasicMuvVariableState{N<:Number}(size::Int, ::Type{N}=Float64) = BasicMuvVariableState{N}(Array(N, size), size)

variate_form{N<:Number}(s::Type{BasicMuvVariableState{N}}) = Multivariate
variate_form{N<:Number}(s::BasicMuvVariableState{N}) = Multivariate

Base.eltype{N<:Number}(::Type{BasicMuvVariableState{N}}) = N
Base.eltype{N<:Number}(s::BasicMuvVariableState{N}) = N

## BasicMavVariableState

type BasicMavVariableState{N<:Number} <: VariableState{Matrixvariate}
  value::Matrix{N}
  size::Tuple{Int, Int}
end

BasicMavVariableState{N<:Number}(value::Matrix{N}) = BasicMavVariableState{N}(value, size(value))

BasicMavVariableState{N<:Number}(size::Tuple, ::Type{N}=Float64) = BasicMavVariableState{N}(Array(N, size...), size)

variate_form{N<:Number}(s::Type{BasicMavVariableState{N}}) = Matrixvariate
variate_form{N<:Number}(s::BasicMavVariableState{N}) = Matrixvariate

Base.eltype{N<:Number}(::Type{BasicMavVariableState{N}}) = N
Base.eltype{N<:Number}(s::BasicMavVariableState{N}) = N
