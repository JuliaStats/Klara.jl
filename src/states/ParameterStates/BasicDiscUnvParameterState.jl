type BasicDiscUnvParameterState{NI<:Integer, NR<:Real} <: ParameterState{Discrete, Univariate}
  value::NI
  loglikelihood::NR
  logprior::NR
  logtarget::NR
  diagnosticvalues::Vector
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

value_support{NI<:Integer, NR<:Real}(s::Type{BasicDiscUnvParameterState{NI, NR}}) = Discrete
value_support{NI<:Integer, NR<:Real}(s::BasicDiscUnvParameterState{NI, NR}) = Discrete

variate_form{NI<:Integer, NR<:Real}(s::Type{BasicDiscUnvParameterState{NI, NR}}) = Univariate
variate_form{NI<:Integer, NR<:Real}(s::BasicDiscUnvParameterState{NI, NR}) = Univariate

Base.eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscUnvParameterState{NI, NR}}) = (NI, NR)
Base.eltype{NI<:Integer, NR<:Real}(s::BasicDiscUnvParameterState{NI, NR}) = (NI, NR)

generate_empty(state::BasicDiscUnvParameterState) =
  BasicDiscUnvParameterState(state.value, state.diagnostickeys, eltype(state)[2])
