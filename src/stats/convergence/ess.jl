### Effective sample size (ESS)

ess(v::AbstractArray; vtype::Symbol=:imse, args...) = length(v)*mcvar_iid(v)/mcvar(v; vtype=vtype, args...)

ess(v::AbstractArray, region; vtype::Symbol=:imse, args...) = mapslices(x -> ess(x; vtype=vtype, args...), v, region)

ess(s::VariableNState{Univariate}; vtype::Symbol=:imse, args...) = ess(s.value; vtype=vtype, args...)

ess(s::VariableNState{Multivariate}, i::Int; vtype::Symbol=:imse, args...) = ess(s.value[i, :]; vtype=vtype, args...)

ess(s::VariableNState{Multivariate}, r::Range=1:s.size; vtype::Symbol=:imse, args...) =
  eltype(s)[ess(s, i; vtype=vtype, args...) for i in r]
