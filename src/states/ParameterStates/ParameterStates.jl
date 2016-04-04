### Abstract parameter state

abstract ParameterState{S<:ValueSupport, F<:VariateForm} <: VariableState{F}

typealias ParameterStateVector{S<:ParameterState} Vector{S}

value_support{S<:ValueSupport, F<:VariateForm}(::Type{ParameterState{S, F}}) = S
variate_form{S<:ValueSupport, F<:VariateForm}(::Type{ParameterState{S, F}}) = F

Base.(:(==)){S<:ParameterState}(z::S, w::S) = reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)])

Base.isequal{S<:ParameterState}(z::S, w::S) = reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)])
