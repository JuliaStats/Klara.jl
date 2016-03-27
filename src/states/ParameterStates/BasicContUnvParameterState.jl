type BasicContUnvParameterState{N<:Real} <: ParameterState{Continuous, Univariate}
  value::N
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::N
  gradlogprior::N
  gradlogtarget::N
  tensorloglikelihood::N
  tensorlogprior::N
  tensorlogtarget::N
  dtensorloglikelihood::N
  dtensorlogprior::N
  dtensorlogtarget::N
  diagnosticvalues::Vector
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
