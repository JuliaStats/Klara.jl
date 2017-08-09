### Abstract variable NStates

abstract type VariableNState{F<:VariateForm} end

add_dimension(n::Number) = eltype(n)[n]
add_dimension(a::Array, sa::Tuple=size(a)) = reshape(a, sa..., 1)

### Basic variable NState subtypes

## BasicUnvVariableNState

mutable struct BasicUnvVariableNState{N<:Number} <: VariableNState{Univariate}
  value::Vector{N}
  n::Integer
end

BasicUnvVariableNState(value::Vector{N}) where {N<:Number} = BasicUnvVariableNState{N}(value, length(value))

BasicUnvVariableNState(n::Integer, ::Type{N}=Float64) where {N<:Number} = BasicUnvVariableNState{N}(Array{N}(n), n)

eltype{N<:Number}(::Type{BasicUnvVariableNState{N}}) = N
eltype{N<:Number}(::BasicUnvVariableNState{N}) = N

copy!(nstate::BasicUnvVariableNState, state::BasicUnvVariableState, i::Integer) = (nstate.value[i] = state.value)

function show(io::IO, nstate::BasicUnvVariableNState{N}) where N<:Number
  indentation = "  "

  println(io, "BasicUnvVariableNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  print(io, indentation*"number of states = $(nstate.n)")
end

## BasicMuvVariableNState

mutable struct BasicMuvVariableNState{N<:Number} <: VariableNState{Multivariate}
  value::Matrix{N}
  size::Integer
  n::Integer
end

BasicMuvVariableNState(value::Matrix{N}) where {N<:Number} = BasicMuvVariableNState{N}(value, size(value)...)

BasicMuvVariableNState(size::Integer, n::Integer, ::Type{N}=Float64) where {N<:Number} =
  BasicMuvVariableNState{N}(Array{N}(size, n), size, n)

eltype{N<:Number}(::Type{BasicMuvVariableNState{N}}) = N
eltype{N<:Number}(::BasicMuvVariableNState{N}) = N

copy!(nstate::BasicMuvVariableNState, state::BasicMuvVariableState, i::Integer) =
  (nstate.value[1+(i-1)*state.size:i*state.size] = state.value)

function show(io::IO, nstate::BasicMuvVariableNState{N}) where N<:Number
  indentation = "  "

  println(io, "BasicMuvVariableNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"state size = $(nstate.size)")
  print(io, indentation*"number of states = $(nstate.n)")
end

## BasicMavVariableNState

mutable struct BasicMavVariableNState{N<:Number} <: VariableNState{Matrixvariate}
  value::Array{N, 3}
  size::Tuple{Integer, Integer}
  n::Integer
end

BasicMavVariableNState(value::Array{N, 3}) where {N<:Number} =
  BasicMavVariableNState{N}(value, (size(value, 1), size(value, 2)), size(value, 3))

BasicMavVariableNState(size::Tuple, n::Integer, ::Type{N}=Float64) where {N<:Number} =
  BasicMavVariableNState{N}(Array{N}(size..., n), size, n)

eltype{N<:Number}(::Type{BasicMavVariableNState{N}}) = N
eltype{N<:Number}(::BasicMavVariableNState{N}) = N

copy!(nstate::BasicMavVariableNState, state::BasicMavVariableState, i::Integer, statelen::Integer=prod(state.size)) =
  (nstate.value[1+(i-1)*statelen:i*statelen] = state.value)

function show(io::IO, nstate::BasicMavVariableNState{N}) where N<:Number
  indentation = "  "

  println(io, "BasicMavVariableNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"state size = $(nstate.size)")
  print(io, indentation*"number of states = $(nstate.n)")
end
