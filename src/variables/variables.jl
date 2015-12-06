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

function codegen_internal_variable_method(f::Function, r::Vector{Symbol}=Symbol[], k::Vector{Symbol}=Symbol[], i::Int=0)
  body = []
  fexprn::Array{Any, 1}

  if isempty(k)
    fexprn = code_lowered(f, (Any,))
  else
    fexprn = code_lowered(f, (Any, Any))
  end

  for exprn in fexprn[1].args[3].args
    if !isa(exprn, LineNumberNode)
      push!(body, exprn)
    end
  end

  fargs = fexprn[1].args[1]
  nfargs = length(fargs)

  for exprn in body
    replace!(exprn, fargs[1], :(_state.value))
  end

  if nfargs > 1
    for exprn in body
      for j in [1:(i-1); (i+1):length(k)]
        replace!(exprn, :($(GlobalRef(Main, :getindex))($(fargs[2]), $(QuoteNode(k[j])))), :(_states[$j].value))
      end
    end
  end

  nr = length(r)

  if nr == 1
    body[end] = Expr(:(=), Expr(:., :_state, QuoteNode(r[1])), body[end].args[1])
  elseif nr > 1
    rvalues = pop!(body)
    rvalues = rvalues.args[1].args
    shift!(rvalues)

    @assert nr == length(rvalues) "Wrong number of returned values in user-defined function"

    for j in 1:nr
      push!(body, Expr(:(=), Expr(:., :_state, QuoteNode(r[j])), rvalues[j]))
    end
  end

  @gensym internal_variable_method

  Expr(:function, Expr(:call, internal_variable_method, :_state, :_states), Expr(:block, body...))
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
