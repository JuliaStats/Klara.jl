### Effective sample size (ESS)

ess{N<:Real}(mcvariance::N, iidvariance::N, len::Integer) = len*iidvariance/mcvariance

ess(v::AbstractArray; vtype::Symbol=:imse, args...) = ess(mcvar(v; vtype=vtype, args...), mcvar_iid(v), length(v))

ess(v::AbstractArray, region; vtype::Symbol=:imse, args...) = mapslices(x -> ess(x; vtype=vtype, args...), v, region)

ess(s::VariableNState{Univariate}; vtype::Symbol=:imse, args...) = ess(s.value; vtype=vtype, args...)

ess(s::VariableNState{Multivariate}, i::Integer; vtype::Symbol=:imse, args...) = ess(s.value[i, :]; vtype=vtype, args...)

ess(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; vtype::Symbol=:imse, args...) =
  eltype(s)[ess(s, i; vtype=vtype, args...) for i in r]
