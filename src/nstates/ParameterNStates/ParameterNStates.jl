### Abstract parameter NStates

abstract ParameterNState{S<:ValueSupport, F<:VariateForm} <: VariableNState{F}

typealias MarkovChain ParameterNState

typealias DiscreteParameterNState{F<:VariateForm} ParameterNState{Discrete, F}
typealias ContinuousParameterNState{F<:VariateForm} ParameterNState{Continuous, F}

typealias UnivariateParameterNState{S<:ValueSupport} ParameterNState{S, Univariate}
typealias MultivariateParameterNState{S<:ValueSupport} ParameterNState{S, Multivariate}
typealias MatrixvariateParameterNState{S<:ValueSupport} ParameterNState{S, Matrixvariate}

diagnostics(nstate::ParameterNState) =
  Dict(zip(nstate.diagnostickeys, Any[nstate.diagnosticvalues[i, :][:] for i = 1:size(nstate.diagnosticvalues, 1)]))

Base.copy!{S<:ValueSupport, F<:VariateForm}(nstate::ParameterNState{S, F}, state::ParameterState{S, F}, i::Integer) =
  nstate.copy(nstate, state, i)
