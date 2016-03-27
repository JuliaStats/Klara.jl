type BasicContMuvParameterState{N<:Real} <: ParameterState{Continuous, Multivariate}
  value::Vector{N}
  loglikelihood::N
  logprior::N
  logtarget::N
  gradloglikelihood::Vector{N}
  gradlogprior::Vector{N}
  gradlogtarget::Vector{N}
  tensorloglikelihood::Matrix{N}
  tensorlogprior::Matrix{N}
  tensorlogtarget::Matrix{N}
  dtensorloglikelihood::Array{N, 3}
  dtensorlogprior::Array{N, 3}
  dtensorlogtarget::Array{N, 3}
  diagnosticvalues::Vector
  size::Int
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

  l = Array(Int, 9)
  for i in 1:9
    l[i] = (monitor[i] == false ? zero(Int) : s)
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
  size::Int,
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContMuvParameterState(fill(convert(N, NaN), size), monitor, diagnostickeys, diagnosticvalues)

BasicContMuvParameterState{N<:Real}(
  size::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Vector=Array(Any, length(diagnostickeys))
) =
  BasicContMuvParameterState(fill(convert(N, NaN), size), monitor, diagnostickeys, diagnosticvalues)

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
