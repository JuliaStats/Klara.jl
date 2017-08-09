### Sampleability

abstract type Sampleability end

mutable struct Deterministic <: Sampleability end

mutable struct Random <: Sampleability end

### Abstract variable

abstract type Variable{S<:Sampleability} end

VariableVector{V<:Variable} = Vector{V}

eltype{S<:Sampleability}(::Type{Variable{S}}) = S

vertex_key(v::Variable) = v.key
vertex_index(v::Variable) = v.index
is_indexed(v::Variable) = v.index > 0 ? true : false

keys(variables::VariableVector) = Symbol[v.key for v in variables]
indexes(variables::VariableVector) = Integer[v.index for v in variables]

sort_by_index(vs::VariableVector) = vs[[v.index for v in vs]]

function default_state(v::Variable, v0, outopts::Dict=Dict())
  local vstate::VariableState

  if isa(v0, VariableState)
    vstate = v0
  elseif isa(v0, Number) ||
    (isa(v0, Vector) && issubtype(eltype(v0), Number)) ||
    (isa(v0, Matrix) && issubtype(eltype(v0), Number))
    if isa(v, Parameter)
      vstate = default_state(v, v0, outopts)
    else
      vstate = default_state(v, v0)
    end
  else
    error("Variable state or state value of type $(typeof(v0)) not valid")
  end

  vstate
end

default_state(v::VariableVector, v0::Vector, outopts::Vector) =
  VariableState[default_state(v[i], v0[i], outopts[i]) for i in 1:length(v0)]

function default_state(v::VariableVector, v0::Vector, outopts::Vector, dpindex::IntegerVector)
  opts = fill(Dict(), length(v0))
  for i in 1:length(dpindex)
    opts[dpindex[i]] = outopts[i]
  end
  default_state(v, v0, opts)
end

show(io::IO, v::Variable) = print(io, "Variable [$(v.index)]: $(v.key) ($(typeof(v)))")

### Deterministic Variable subtypes

## Constant

mutable struct Constant <: Variable{Deterministic}
  key::Symbol
  index::Integer
end

Constant(key::Symbol) = Constant(key, 0)

default_state(variable::Constant, value::N) where {N<:Number} = BasicUnvVariableState(value)
default_state(variable::Constant, value::Vector{N}) where {N<:Number} = BasicMuvVariableState(value)
default_state(variable::Constant, value::Matrix{N}) where {N<:Number} = BasicMavVariableState(value)

show(io::IO, ::Type{Constant}) = print(io, "Constant")

dotshape(variable::Constant) = "trapezium"

## Hyperparameter

const Hyperparameter = Constant

## Data

mutable struct Data <: Variable{Deterministic}
  key::Symbol
  index::Integer
  update::Union{Function, Void}
end

Data(key::Symbol, index::Integer) = Data(key, index, nothing)
Data(key::Symbol, update::Union{Function, Void}) = Data(key, 0, update)
Data(key::Symbol) = Data(key, 0, nothing)

default_state(variable::Data, value::N) where {N<:Number} = BasicUnvVariableState(value)
default_state(variable::Data, value::Vector{N}) where {N<:Number} = BasicMuvVariableState(value)
default_state(variable::Data, value::Matrix{N}) where {N<:Number} = BasicMavVariableState(value)

show(io::IO, ::Type{Data}) = print(io, "Data")

dotshape(variable::Data) = "box"

## Transformation

mutable struct Transformation <: Variable{Deterministic}
  key::Symbol
  index::Integer
  transform::Function
  states::VariableStateVector
end

Transformation(key::Symbol, index::Integer, transform::Function=()->()) =
  Transformation(key, index, transform, VariableState[])

Transformation(key::Symbol, transform::Function=()->(), states::VariableStateVector=VariableState[]) =
  Transformation(key, 0, transform, states)

default_state(variable::Transformation, value::N) where {N<:Number} = BasicUnvVariableState(value)
default_state(variable::Transformation, value::Vector{N}) where {N<:Number} = BasicMuvVariableState(value)
default_state(variable::Transformation, value::Matrix{N}) where {N<:Number} = BasicMavVariableState(value)

show(io::IO, ::Type{Transformation}) = print(io, "Transformation")

dotshape(variable::Transformation) = "polygon"
