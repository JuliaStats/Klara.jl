### Compute mean recursively given previous mean and new sample

mean!(lastmean::RealVector, k::Integer, x::RealVector) = (lastmean[:] = ((k-1)*lastmean+x)/k)

mean(s::VariableNState{Univariate}) = mean(s.value)

mean(s::VariableNState{Multivariate}, i::Integer) = mean(s.value[i, :])

mean(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size) = eltype(s)[mean(s, i) for i in r]
