### Abstract parameter NStates

abstract type ParameterNState{S<:ValueSupport, F<:VariateForm} <: VariableNState{F} end

const MarkovChain = ParameterNState

DiscreteParameterNState{F<:VariateForm} = ParameterNState{Discrete, F}
ContinuousParameterNState{F<:VariateForm} = ParameterNState{Continuous, F}

UnivariateParameterNState{S<:ValueSupport} = ParameterNState{S, Univariate}
MultivariateParameterNState{S<:ValueSupport} = ParameterNState{S, Multivariate}
MatrixvariateParameterNState{S<:ValueSupport} = ParameterNState{S, Matrixvariate}

diagnostics(nstate::ParameterNState) =
  Dict(zip(nstate.diagnostickeys, Any[nstate.diagnosticvalues[i, :][:] for i = 1:size(nstate.diagnosticvalues, 1)]))

copy!(nstate::ParameterNState{S, F}, state::ParameterState{S, F}, i::Integer) where {S<:ValueSupport, F<:VariateForm} =
  nstate.copy(nstate, state, i)
