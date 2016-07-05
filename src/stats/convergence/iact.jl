### Integrated autocorrelation time (IACT)

iact{N<:Real}(mcvariance::N, iidvariance::N) = mcvariance/iidvariance

iact(v::AbstractArray; vtype::Symbol=:imse, args...) = iact(mcvar(v; vtype=vtype, args...), mcvar_iid(v))

iact(v::AbstractArray, region; vtype::Symbol=:imse, args...) = mapslices(x -> iact(x; vtype=vtype, args...), v, region)

iact(s::VariableNState{Univariate}; vtype::Symbol=:imse, args...) = iact(s.value; vtype=vtype, args...)

iact(s::VariableNState{Multivariate}, i::Integer; vtype::Symbol=:imse, args...) = iact(s.value[i, :]; vtype=vtype, args...)

iact(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; vtype::Symbol=:imse, args...) =
  eltype(s)[iact(s, i; vtype=vtype, args...) for i in r]
