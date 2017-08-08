"""
abstract ParameterState{S<:ValueSupport, F<:VariateForm} <: VariableState{F}

Root of parameter state type hierarchy
"""
abstract type ParameterState{S<:ValueSupport, F<:VariateForm} <: VariableState{F} end

DiscreteParameterState{F<:VariateForm} = ParameterState{Discrete, F}
ContinuousParameterState{F<:VariateForm} = ParameterState{Continuous, F}

UnivariateParameterState{S<:ValueSupport} = ParameterState{S, Univariate}
MultivariateParameterState{S<:ValueSupport} = ParameterState{S, Multivariate}
MatrixvariateParameterState{S<:ValueSupport} = ParameterState{S, Matrixvariate}

ParameterStateVector{S<:ParameterState} = Vector{S}

value_support{S<:ValueSupport, F<:VariateForm}(::Type{ParameterState{S, F}}) = S
variate_form{S<:ValueSupport, F<:VariateForm}(::Type{ParameterState{S, F}}) = F

diagnostics(state::ParameterState) = Dict(zip(state.diagnostickeys, state.diagnosticvalues))

==(z::S, w::S) where {S<:ParameterState} = reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)])

isequal(z::S, w::S) where {S<:ParameterState} = reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)])
