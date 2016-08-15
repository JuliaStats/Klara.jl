### Compute mean recursively given previous mean and new sample

Base.mean!(lastmean::RealVector, k::Integer, x::RealVector) = (lastmean[:] = ((k-1)*lastmean+x)/k)

Base.mean(s::VariableNState{Univariate}) = mean(s.value)

Base.mean(s::VariableNState{Multivariate}, i::Integer) = mean(s.value[i, :])

Base.mean(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size) = eltype(s)[mean(s, i) for i in r]
