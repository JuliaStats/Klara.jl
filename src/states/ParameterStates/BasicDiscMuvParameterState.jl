type BasicDiscMuvParameterState{NI<:Integer, NR<:Real} <: ParameterState{Discrete, Multivariate}
  value::Vector{NI}
  loglikelihood::NR
  logprior::NR
  logtarget::NR
  diagnosticvalues::Vector
  size::Int
  diagnostickeys::Vector{Symbol}
end

function BasicDiscMuvParameterState{NI<:Integer, NR<:Real}(
  value::Vector{NI},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NR}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
)
  v = convert(NR, NaN)
  BasicDiscMuvParameterState{NI, NR}(value, v, v, v, diagnosticvalues, length(value), diagnostickeys)
end

BasicDiscMuvParameterState{NI<:Integer, NR<:Real}(
  size::Int,
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int,
  ::Type{NR}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicDiscMuvParameterState(Array(NI, size), diagnostickeys, NR, diagnosticvalues)

value_support{NI<:Integer, NR<:Real}(s::Type{BasicDiscMuvParameterState{NI, NR}}) = Discrete
value_support{NI<:Integer, NR<:Real}(s::BasicDiscMuvParameterState{NI, NR}) = Discrete

variate_form{NI<:Integer, NR<:Real}(s::Type{BasicDiscMuvParameterState{NI, NR}}) = Univariate
variate_form{NI<:Integer, NR<:Real}(s::BasicDiscMuvParameterState{NI, NR}) = Univariate

Base.eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscMuvParameterState{NI, NR}}) = (NI, NR)
Base.eltype{NI<:Integer, NR<:Real}(s::BasicDiscMuvParameterState{NI, NR}) = (NI, NR)

generate_empty(state::BasicDiscMuvParameterState) =
  BasicDiscMuvParameterState(state.size, state.diagnostickeys, eltype(state)...)
