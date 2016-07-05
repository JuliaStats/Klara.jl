Base.mean(s::VariableNState{Univariate}) = mean(s.value)

Base.mean(s::VariableNState{Multivariate}, i::Integer) = mean(s.value[i, :])

Base.mean(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size) = eltype(s)[mean(s, i) for i in r]
