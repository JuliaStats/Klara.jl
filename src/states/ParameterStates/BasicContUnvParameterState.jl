"""
Basic continuous univariate parameter state type

# Constructors

## BasicContUnvParameterState{N<:Real}(value::N, <optional arguments>)

Construct a basic continuous univariate parameter state with some ``value``.

### Optional arguments:
  
* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

## BasicContUnvParameterState{N<:Real}(<optional arguments>)

Construct a basic continuous univariate parameter state with an uninitialized ``value`` (``NaN``).

### Optional arguments:
  
* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{N}=Float64``: the element type of the state value.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

# Examples

```julia
julia> state = BasicContUnvParameterState(-1.25, [:accept], [false])
Lora.BasicContUnvParameterState{Float64}(-1.25,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,Bool[false],[:accept])

julia> state.value
-1.25

julia> diagnostics(state)
Dict{Symbol,Bool} with 1 entry:
  :accept => false
```
"""
type BasicContUnvParameterState{N<:Real} <: ParameterState{Continuous, Univariate}
  "Vector value of basic continuous univariate parameter state"
  value::N
  "Value of log-likelihood at the state's value"
  loglikelihood::N
  "Value of log-prior at the state's value"
  logprior::N
  "Value of log-target at the state's value"
  logtarget::N
  "Value of gradient of log-likelihood at the state's value"
  gradloglikelihood::N
  "Value of gradient of log-prior at the state's value"
  gradlogprior::N
  "Value of gradient of log-target at the state's value"
  gradlogtarget::N
  "Value of metric tensor of log-likelihood at the state's value"
  tensorloglikelihood::N
  "Value of metric tensor of log-prior at the state's value"
  tensorlogprior::N
  "Value of metric tensor of log-target at the state's value"
  tensorlogtarget::N
  "Value of derivatives of metric tensor of log-likelihood at the state's value"
  dtensorloglikelihood::N
  "Value of derivatives of metric tensor of log-prior at the state's value"
  dtensorlogprior::N
  "Value of derivatives of metric tensor of log-target at the state's value"
  dtensorlogtarget::N
  "Diagnostic values associated with the sampling of the state"
  diagnosticvalues::Vector
  "Diagnostic keys associated with the sampling of the state"
  diagnostickeys::Vector{Symbol}
end

function BasicContUnvParameterState{N<:Real}(
  value::N,
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(N, NaN)
  BasicContUnvParameterState{N}(value, v, v, v, v, v, v, v, v, v, v, v, v, diagnosticvalues, diagnostickeys)
end

BasicContUnvParameterState{N<:Real}(
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContUnvParameterState(convert(N, NaN), diagnostickeys, diagnosticvalues)

value_support{N<:Real}(::Type{BasicContUnvParameterState{N}}) = Continuous
value_support(::BasicContUnvParameterState) = Continuous

variate_form{N<:Real}(::Type{BasicContUnvParameterState{N}}) = Univariate
variate_form(::BasicContUnvParameterState) = Univariate

Base.eltype{N<:Real}(::Type{BasicContUnvParameterState{N}}) = N
Base.eltype{N<:Real}(::BasicContUnvParameterState{N}) = N

generate_empty(state::BasicContUnvParameterState) = BasicContUnvParameterState(state.diagnostickeys, eltype(state))
