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
Klara.BasicContUnvParameterState{Float64}(-1.25,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,Bool[false],[:accept])

julia> state.value
-1.25

julia> diagnostics(state)
Dict{Symbol,Bool} with 1 entry:
  :accept => false
```
"""
mutable struct BasicContUnvParameterState{N<:Real} <: ParameterState{Continuous, Univariate}
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
  diffmethods::Union{DiffMethods, Void}
  diffstate::Union{DiffState, Void}
end

function BasicContUnvParameterState(
  value::N,
  diagnostickeys::Vector{Symbol}=Symbol[],
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))
) where N<:Real
  v = convert(N, NaN)

  diffstate = DiffState()

  if diffopts != nothing
    if diffopts.order == 1
      for (i, field) in ((1, :resultll), (2, :resultlp), (3, :resultlt))
        if diffopts.targets[i]
          setfield!(diffstate, field, DiffResults.DiffResult(zero(value), zero(value)))
        end
      end
    else
      error("Derivative order must be 1, got order=$order")
    end
  end

  BasicContUnvParameterState{N}(
    value, v, v, v, v, v, v, v, v, v, v, v, v, diagnosticvalues, diagnostickeys, diffmethods, diffstate
  )
end

BasicContUnvParameterState(
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))
) where {N<:Real} =
  BasicContUnvParameterState(convert(N, NaN), diagnostickeys, diffmethods, diffopts, diagnosticvalues)

value_support{N<:Real}(::Type{BasicContUnvParameterState{N}}) = Continuous
value_support(::BasicContUnvParameterState) = Continuous

variate_form{N<:Real}(::Type{BasicContUnvParameterState{N}}) = Univariate
variate_form(::BasicContUnvParameterState) = Univariate

eltype{N<:Real}(::Type{BasicContUnvParameterState{N}}) = N
eltype{N<:Real}(::BasicContUnvParameterState{N}) = N

generate_empty(
  state::BasicContUnvParameterState,
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing
) = BasicContUnvParameterState(state.diagnostickeys, eltype(state), diffmethods, diffopts)
