### Abstract variable NStates

abstract VariableNState{F<:VariateForm}

add_dimension(n::Number) = eltype(n)[n]
add_dimension(a::Array, sa::Tuple=size(a)) = reshape(a, sa..., 1)

Base.writemime(io::IO, ::MIME"text/plain", nstate::VariableNState) = show(io, nstate)

### Basic variable NState subtypes

## BasicUnvVariableNState

type BasicUnvVariableNState{N<:Number} <: VariableNState{Univariate}
  value::Vector{N}
  n::Integer
end

BasicUnvVariableNState{N<:Number}(value::Vector{N}) = BasicUnvVariableNState{N}(value, length(value))

BasicUnvVariableNState{N<:Number}(n::Integer, ::Type{N}=Float64) = BasicUnvVariableNState{N}(Array(N, n), n)

Base.eltype{N<:Number}(::Type{BasicUnvVariableNState{N}}) = N
Base.eltype{N<:Number}(::BasicUnvVariableNState{N}) = N

Base.copy!(nstate::BasicUnvVariableNState, state::BasicUnvVariableState, i::Integer) = (nstate.value[i] = state.value)

function Base.show{N<:Number}(io::IO, nstate::BasicUnvVariableNState{N})
  indentation = "  "

  println(io, "BasicUnvVariableNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  print(io, indentation*"number of states = $(nstate.n)")
end

Base.writemime{N<:Number}(io::IO, ::MIME"text/plain", nstate::BasicUnvVariableNState{N}) = show(io, nstate)

## BasicMuvVariableNState

type BasicMuvVariableNState{N<:Number} <: VariableNState{Multivariate}
  value::Matrix{N}
  size::Integer
  n::Integer
end

BasicMuvVariableNState{N<:Number}(value::Matrix{N}) = BasicMuvVariableNState{N}(value, size(value)...)

BasicMuvVariableNState{N<:Number}(size::Integer, n::Integer, ::Type{N}=Float64) =
  BasicMuvVariableNState{N}(Array(N, size, n), size, n)

Base.eltype{N<:Number}(::Type{BasicMuvVariableNState{N}}) = N
Base.eltype{N<:Number}(::BasicMuvVariableNState{N}) = N

Base.copy!(nstate::BasicMuvVariableNState, state::BasicMuvVariableState, i::Integer) =
  (nstate.value[1+(i-1)*state.size:i*state.size] = state.value)

function Base.show{N<:Number}(io::IO, nstate::BasicMuvVariableNState{N})
  indentation = "  "

  println(io, "BasicMuvVariableNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"state size = $(nstate.size)")
  print(io, indentation*"number of states = $(nstate.n)")
end

Base.writemime{N<:Number}(io::IO, ::MIME"text/plain", nstate::BasicMuvVariableNState{N}) = show(io, nstate)

## BasicMavVariableNState

type BasicMavVariableNState{N<:Number} <: VariableNState{Matrixvariate}
  value::Array{N, 3}
  size::Tuple{Integer, Integer}
  n::Integer
end

BasicMavVariableNState{N<:Number}(value::Array{N, 3}) =
  BasicMavVariableNState{N}(value, (size(value, 1), size(value, 2)), size(value, 3))

BasicMavVariableNState{N<:Number}(size::Tuple, n::Integer, ::Type{N}=Float64) =
  BasicMavVariableNState{N}(Array(N, size..., n), size, n)

Base.eltype{N<:Number}(::Type{BasicMavVariableNState{N}}) = N
Base.eltype{N<:Number}(::BasicMavVariableNState{N}) = N

Base.copy!(nstate::BasicMavVariableNState, state::BasicMavVariableState, i::Integer, statelen::Integer=prod(state.size)) =
  (nstate.value[1+(i-1)*statelen:i*statelen] = state.value)

function Base.show{N<:Number}(io::IO, nstate::BasicMavVariableNState{N})
  indentation = "  "

  println(io, "BasicMavVariableNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"state size = $(nstate.size)")
  print(io, indentation*"number of states = $(nstate.n)")
end
