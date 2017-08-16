"""
Basic discrete multivariate parameter state type

# Constructors

## BasicDiscMuvParameterState{NI<:Integer, NR<:Real}(value::Vector{NI}, <optional arguments>)

Construct a basic discrete multivariate parameter state with some ``value``.

###  Optional arguments:

* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{NR}=Float64``: the element type of target-related fields.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

## BasicDiscMuvParameterState{NI<:Integer, NR<:Real}(size::Integer, <optional arguments>)

Construct a basic discrete multivariate parameter state with with a ``value`` of specified ``size``.

###  Optional arguments:

* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{NI}=Integer``: the element type of the state value.
* ``::Type{NR}=Float64``: the element type of target-related fields.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

# Examples

```julia
julia> state = BasicDiscMuvParameterState(Int64[0, 1], [:accept], Float64, [false])
Klara.BasicDiscMuvParameterState{Int64,Float64}([0,1],NaN,NaN,NaN,Bool[false],2,[:accept])

julia> state.value
2-element Array{Int64,1}:
 0
 1

julia> diagnostics(state)
Dict{Symbol,Bool} with 1 entry:
  :accept => false
```
"""
mutable struct BasicDiscMuvParameterState{NI<:Integer, NR<:Real} <: ParameterState{Discrete, Multivariate}
  "Vector value of basic discrete multivariate parameter state"
  value::Vector{NI}
  "Value of log-likelihood at the state's value"
  loglikelihood::NR
  "Value of log-prior at the state's value"
  logprior::NR
  "Value of log-target at the state's value"
  logtarget::NR
  "Diagnostic values associated with the sampling of the state"
  diagnosticvalues::Vector
  "Integer representing the length of vector value of basic discrete multivariate parameter state"
  size::Integer
  "Diagnostic keys associated with the sampling of the state"
  diagnostickeys::Vector{Symbol}
end

function BasicDiscMuvParameterState(
  value::Vector{NI},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NR}=Float64,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))
) where {NI<:Integer, NR<:Real}
  v = convert(NR, NaN)
  BasicDiscMuvParameterState{NI, NR}(value, v, v, v, diagnosticvalues, length(value), diagnostickeys)
end

BasicDiscMuvParameterState(
  size::Integer,
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))
) where {NI<:Integer, NR<:Real} =
  BasicDiscMuvParameterState(Array{NI}(size), diagnostickeys, NR, diagnosticvalues)

value_support{NI<:Integer, NR<:Real}(::Type{BasicDiscMuvParameterState{NI, NR}}) = Discrete
value_support(::BasicDiscMuvParameterState) = Discrete

variate_form{NI<:Integer, NR<:Real}(::Type{BasicDiscMuvParameterState{NI, NR}}) = Univariate
variate_form(::BasicDiscMuvParameterState) = Univariate

eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscMuvParameterState{NI, NR}}) = (NI, NR)
eltype{NI<:Integer, NR<:Real}(::BasicDiscMuvParameterState{NI, NR}) = (NI, NR)

generate_empty(state::BasicDiscMuvParameterState) =
  BasicDiscMuvParameterState(state.size, state.diagnostickeys, eltype(state)...)
