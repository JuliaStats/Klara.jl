### Compute mean recursively given previous mean and new sample

recursive_mean(lastmean::Real, k::Integer, x::Real) = ((k-1)*lastmean+x)/k

recursive_mean!(m::RealVector, lastmean::RealVector, k::Integer, x::RealVector) = (m[:] = ((k-1)*lastmean+x)/k)

mean(s::VariableNState{Univariate}) = mean(s.value)

mean(s::VariableNState{Multivariate}, i::Integer) = mean(s.value[i, :])

mean(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size) = eltype(s)[mean(s, i) for i in r]
