### Integrated autocorrelation time (IACT)

iact(mcvariance::N, iidvariance::N) where {N<:Real} = mcvariance/iidvariance

iact(v::AbstractVector, vtype::Symbol, args...) = iact(mcvar(v, Val{vtype}, args...), mcvar(v, Val{:iid}))

iact(v::AbstractArray, vtype::Symbol, region, args...) = mapslices(x -> iact(vec(x), vtype, args...), v, region)

iact(s::VariableNState{Univariate}, vtype::Symbol, args...) = iact(s.value, vtype, args...)

iact(s::VariableNState{Multivariate}, vtype::Symbol, i::Integer, args...) = iact(s.value[i, :], vtype, args...)

iact(s::VariableNState{Multivariate}, vtype::Symbol, r::AbstractVector=1:s.size, args...) =
  eltype(s)[iact(s, vtype, i, args...) for i in r]

iact(v::AbstractVector, args...) = iact(v, :imse, args...)

iact(v::AbstractArray, region, args...) = iact(v, :imse, region, args...)

iact(s::VariableNState{Univariate}, args...) = iact(s.value, :imse, args...)

iact(s::VariableNState{Multivariate}, i::Integer, args...) = iact(s, :imse, i, args...)

iact(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size, args...) = iact(s, :imse, r, args...)
