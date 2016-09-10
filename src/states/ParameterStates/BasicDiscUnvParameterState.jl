"""
Basic discrete univariate parameter state type

# Constructors

## BasicDiscUnvParameterState{NI<:Integer, NR<:Real}(value::NI, <optional arguments>)

Construct a basic discrete univariate parameter state with some ``value``.

### Optional arguments:
  
* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{NR}=Float64``: the element type of target-related fields.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

# Examples

```julia
julia> state = BasicDiscUnvParameterState(2, [:accept], Float64, [true])
Klara.BasicDiscUnvParameterState{Int64,Float64}(2,NaN,NaN,NaN,Bool[true],[:accept])

julia> state.value
2

julia> diagnostics(state)
Dict{Symbol,Bool} with 1 entry:
  :accept => true
```
"""
type BasicDiscUnvParameterState{NI<:Integer, NR<:Real} <: ParameterState{Discrete, Univariate}
  "Scalar value of basic discrete univariate parameter state"
  value::NI
  "Value of log-likelihood at the state's value"
  loglikelihood::NR
  "Value of log-prior at the state's value"
  logprior::NR
  "Value of log-target at the state's value"
  logtarget::NR
  "Diagnostic values associated with the sampling of the state"
  diagnosticvalues::Vector
  "Diagnostic keys associated with the sampling of the state"
  diagnostickeys::Vector{Symbol}
end

function BasicDiscUnvParameterState{NI<:Integer, NR<:Real}(
  value::NI,
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NR}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(NR, NaN)
  BasicDiscUnvParameterState{NI, NR}(value, v, v, v, diagnosticvalues, diagnostickeys)
end

value_support{NI<:Integer, NR<:Real}(::Type{BasicDiscUnvParameterState{NI, NR}}) = Discrete
value_support(::BasicDiscUnvParameterState) = Discrete

variate_form{NI<:Integer, NR<:Real}(::Type{BasicDiscUnvParameterState{NI, NR}}) = Univariate
variate_form(::BasicDiscUnvParameterState) = Univariate

Base.eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscUnvParameterState{NI, NR}}) = (NI, NR)
Base.eltype{NI<:Integer, NR<:Real}(::BasicDiscUnvParameterState{NI, NR}) = (NI, NR)

generate_empty(state::BasicDiscUnvParameterState) =
  BasicDiscUnvParameterState(state.value, state.diagnostickeys, eltype(state)[2])
