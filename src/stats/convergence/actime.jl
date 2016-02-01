### Integrated autocorrelation time

actime(v::AbstractArray; vtype::Symbol=:imse, args...) = mcvar(v; vtype=vtype, args...)/mcvar_iid(v)

actime(v::AbstractArray, region; vtype::Symbol=:imse, args...) = mapslices(x -> actime(x; vtype=vtype, args...), v, region)

actime(s::VariableNState{Univariate}; vtype::Symbol=:imse, args...) = actime(s.value; vtype=vtype, args...)

actime(s::VariableNState{Multivariate}, i::Int; vtype::Symbol=:imse, args...) = actime(s.value[i, :]; vtype=vtype, args...)

actime(s::VariableNState{Multivariate}, r::Range=1:s.size; vtype::Symbol=:imse, args...) =
  eltype(s)[actime(s, i; vtype=vtype, args...) for i in r]
