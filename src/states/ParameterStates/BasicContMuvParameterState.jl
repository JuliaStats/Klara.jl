"""
Basic continuous multivariate parameter state type

# Constructors

## BasicContMuvParameterState{N<:Real}(value::Vector{N}, <optional arguments>)

Construct a basic continuous multivariate parameter state with some ``value``.

###  Optional arguments:

* ``monitor::Vector{Bool}=fill(false, 9)``: 9-element Boolean vector indicating which of the target-related fields are
stored by the state.
* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

## BasicContMuvParameterState{N<:Real}(value::Vector{N}, monitor::Vector{Symbol}, <optional arguments>)

Construct a basic continuous multivariate parameter state with some ``value`` and tracked target-related fields specified by
``monitor``.

###  Optional arguments:

* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

## BasicContMuvParameterState{N<:Real}(size::Integer, <optional arguments>)

Construct a basic continuous multivariate parameter state with a ``value`` of specified ``size``.

###  Optional arguments:

* ``monitor::Vector{Bool}=fill(false, 9)``: 9-element Boolean vector indicating which of the target-related fields are
stored by the state.
* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{N}=Float64``: the element type of the state value.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

## BasicContMuvParameterState{N<:Real}(size::Integer, monitor::Vector{Symbol}, <optional arguments>)

Construct a basic continuous multivariate parameter state with a ``value`` of specified ``size`` and tracked target-related
fields specified by ``monitor``.

###  Optional arguments:

* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{N}=Float64``: the element type of the state value.
* ``diagnosticvalues::Vector=Array(Any, length(diagnostickeys))``: the diagnostic values of the state.

# Examples

```julia
julia> state = BasicContMuvParameterState(zeros(Float64, 2), [:logtarget])
Lora.BasicContMuvParameterState{Float64}([0.0,0.0],NaN,NaN,NaN,Float64[],Float64[],Float64[],0x0 Array{Float64,2},0x0 Array{Float64,2},0x0 Array{Float64,2},0x0x0 Array{Float64,3},0x0x0 Array{Float64,3},0x0x0 Array{Float64,3},Any[],2,Symbol[])

julia> state.value
2-element Array{Float64,1}:
 0.0
 0.0
```
"""
type BasicContMuvParameterState{N<:Real} <: ParameterState{Continuous, Multivariate}
  "Vector value of basic continuous multivariate parameter state"
  value::Vector{N}
  "Value of log-likelihood at the state's value"
  loglikelihood::N
  "Value of log-prior at the state's value"
  logprior::N
  "Value of log-target at the state's value"
  logtarget::N
  "Value of gradient of log-likelihood at the state's value"
  gradloglikelihood::Vector{N}
  "Value of gradient of log-prior at the state's value"
  gradlogprior::Vector{N}
  "Value of gradient of log-target at the state's value"
  gradlogtarget::Vector{N}
  "Value of metric tensor of log-likelihood at the state's value"
  tensorloglikelihood::Matrix{N}
  "Value of metric tensor of log-prior at the state's value"
  tensorlogprior::Matrix{N}
  "Value of metric tensor of log-target at the state's value"
  tensorlogtarget::Matrix{N}
  "Value of derivatives of metric tensor of log-likelihood at the state's value"
  dtensorloglikelihood::Array{N, 3}
  "Value of derivatives of metric tensor of log-prior at the state's value"
  dtensorlogprior::Array{N, 3}
  "Value of derivatives of metric tensor of log-target at the state's value"
  dtensorlogtarget::Array{N, 3}
  "Diagnostic values associated with the sampling of the state"
  diagnosticvalues::Vector
  "Integer representing the length of vector value of basic continuous multivariate parameter state"
  size::Integer
  "Diagnostic keys associated with the sampling of the state"
  diagnostickeys::Vector{Symbol}
end

function BasicContMuvParameterState{N<:Real}(
  value::Vector{N},
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(N, NaN)

  s = length(value)

  l = Array(Integer, 9)
  for i in 1:9
    l[i] = (monitor[i] == false ? zero(Integer) : s)
  end

  BasicContMuvParameterState{N}(
    value,
    v,
    v,
    v,
    Array(N, l[1]),
    Array(N, l[2]),
    Array(N, l[3]),
    Array(N, l[4], l[4]),
    Array(N, l[5], l[5]),
    Array(N, l[6], l[6]),
    Array(N, l[7], l[7], l[7]),
    Array(N, l[8], l[8], l[8]),
    Array(N, l[9], l[9], l[9]),
    diagnosticvalues,
    s,
    diagnostickeys
  )
end

function BasicContMuvParameterState{N<:Real}(
  value::Vector{N},
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  fnames = fieldnames(BasicContMuvParameterState)
  BasicContMuvParameterState(value, [fnames[i] in monitor ? true : false for i in 5:13], diagnostickeys, diagnosticvalues)
end

BasicContMuvParameterState{N<:Real}(
  size::Integer,
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContMuvParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

BasicContMuvParameterState{N<:Real}(
  size::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContMuvParameterState(Array(N, size), monitor, diagnostickeys, diagnosticvalues)

value_support{N<:Real}(::Type{BasicContMuvParameterState{N}}) = Continuous
value_support(::BasicContMuvParameterState) = Continuous

variate_form{N<:Real}(::Type{BasicContMuvParameterState{N}}) = Multivariate
variate_form(::BasicContMuvParameterState) = Multivariate

Base.eltype{N<:Real}(::Type{BasicContMuvParameterState{N}}) = N
Base.eltype{N<:Real}(::BasicContMuvParameterState{N}) = N

generate_empty(
  state::BasicContMuvParameterState,
  monitor::Vector{Bool}=[isempty(getfield(state, fieldnames(BasicContMuvParameterState)[i])) ? false : true for i in 5:13]
) =
  BasicContMuvParameterState(state.size, monitor, state.diagnostickeys, eltype(state))
