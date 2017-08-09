### Effective sample size (ESS)

ess(mcvariance::N, iidvariance::N, len::Integer) where {N<:Real} = len*iidvariance/mcvariance

ess(v::AbstractVector, vtype::Symbol, args...) = ess(mcvar(v, Val{vtype}, args...), mcvar(v, Val{:iid}), length(v))

ess(v::AbstractArray, vtype::Symbol, region, args...) = mapslices(x -> ess(vec(x), vtype, args...), v, region)

ess(s::VariableNState{Univariate}, vtype::Symbol, args...) = ess(s.value, vtype, args...)

ess(s::VariableNState{Multivariate}, vtype::Symbol, i::Integer, args...) = ess(s.value[i, :], vtype, args...)

ess(s::VariableNState{Multivariate}, vtype::Symbol, r::AbstractVector=1:s.size, args...) =
  eltype(s)[ess(s, vtype, i, args...) for i in r]

ess(v::AbstractVector, args...) = ess(v, :imse, args...)

ess(v::AbstractArray, region, args...) = ess(v, :imse, region, args...)

ess(s::VariableNState{Univariate}, args...) = ess(s.value, :imse, args...)

ess(s::VariableNState{Multivariate}, i::Integer, args...) = ess(s, :imse, i, args...)

ess(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size, args...) = ess(s, :imse, r, args...)
