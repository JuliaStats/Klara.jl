### Sampleability

abstract Sampleability

type Deterministic <: Sampleability end

type Random <: Sampleability end

### Variable

abstract Variable{S<:Sampleability}

Base.eltype{S<:Sampleability}(::Type{Variable{S}}) = S

vertex_index(v::Variable) = v.index

is_indexed(v::Variable) = v.index > 0 ? true : false

Base.convert(::Type{KeyVertex}, v::Variable) = KeyVertex{Symbol}(v.index, v.key)
Base.convert(::Type{Vector{KeyVertex}}, v::Vector{Variable}) = KeyVertex{Symbol}[convert(KeyVertex, i) for i in v]

function codegen_internal_variable_method(f::Function, r::Vector{Symbol}, nkeys::Int=0, nfargs::Bool=true)
  fargs::Union{Expr, Vector}
  rvalues::Expr

  if nkeys == 0
    fargs = [:(_state.value)]
  elseif nkeys > 0
    if nfargs
      fargs = [Expr(:ref, :Any, [:(_states[$j].value) for j in 1:nkeys]...)]
    else
      fargs = [:(_state.value), Expr(:ref, :Any, [:(_states[$j].value) for j in 1:nkeys]...)]
    end
  else
    error("nkeys must be positive")
  end

  nr = length(r)
  if nr == 1
    rvalues = Expr(:., :_state, QuoteNode(r[1]))
  elseif nr > 1
    rvalues = Expr(:tuple, [Expr(:., :_state, QuoteNode(r[j])) for j in 1:nr]...)
  else
    error("Vector of return symbols must have at leasat one element")
  end

  body = Expr(:(=), rvalues, :($(f)($(fargs...))))

  @gensym internal_variable_method

  quote
    function $internal_variable_method(_state, _states)
      $(body)
    end
  end
end

Base.show(io::IO, v::Variable) = print(io, "Variable [$(v.index)]: $(v.key) ($(typeof(v)))")

### Deterministic Variable subtypes

## Constant

immutable Constant <: Variable{Deterministic}
  key::Symbol
  index::Int
end

Constant(key::Symbol) = Constant(key, 0)

default_state{N<:Number}(variable::Constant, value::N) = BasicUnvVariableState(value)
default_state{N<:Number}(variable::Constant, value::Vector{N}) = BasicMuvVariableState(value)
default_state{N<:Number}(variable::Constant, value::Matrix{N}) = BasicMavVariableState(value)

## Hyperparameter

typealias Hyperparameter Constant

## Data

immutable Data <: Variable{Deterministic}
  key::Symbol
  index::Int
  update::Union{Function, Void}
end

Data(key::Symbol, index::Int) = Data(key, index, nothing)
Data(key::Symbol, update::Union{Function, Void}) = Data(key, 0, update)
Data(key::Symbol) = Data(key, 0, nothing)

default_state{N<:Number}(variable::Data, value::N) = BasicUnvVariableState(value)
default_state{N<:Number}(variable::Data, value::Vector{N}) = BasicMuvVariableState(value)
default_state{N<:Number}(variable::Data, value::Matrix{N}) = BasicMavVariableState(value)

## Transformation

immutable Transformation <: Variable{Deterministic}
  key::Symbol
  index::Int
  transform::Function
end

Transformation(key::Symbol, transform::Function) = Transformation(key, 0, transform)

default_state{N<:Number}(variable::Transformation, value::N) = BasicUnvVariableState(value)
default_state{N<:Number}(variable::Transformation, value::Vector{N}) = BasicMuvVariableState(value)
default_state{N<:Number}(variable::Transformation, value::Matrix{N}) = BasicMavVariableState(value)

### Random Variable subtypes

## Abstract Parameter types

abstract Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random}

value_support{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = S
variate_form{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = F
