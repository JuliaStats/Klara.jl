### Abstract parameter NStates

abstract ParameterNState{S<:ValueSupport, F<:VariateForm} <: VariableNState{F}

typealias MarkovChain ParameterNState

diagnostics(nstate::ParameterNState) =
  Dict(zip(nstate.diagnostickeys, Any[nstate.diagnosticvalues[i, :][:] for i = 1:size(nstate.diagnosticvalues, 1)]))
