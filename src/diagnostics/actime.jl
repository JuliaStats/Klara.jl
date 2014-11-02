### Integrated autocorrelation time

function actime(x::Vector{Float64}; vtype::Symbol=:imse, args...)
  @assert in(vtype, actypes) "Unknown actime type $vtype"

  return mcvar(x; vtype=vtype, args...)/mcvar(x; vtype=:iid)
end

actime(x::Matrix{Float64}, pars::Ranges=1:size(x, 2); vtype::Symbol=:imse, args...) =
  Float64[actime(x[:, pars[i]]; vtype=vtype, args...) for i = 1:length(pars)]

actime(x::Matrix{Float64}, par::Real; vtype::Symbol=:imse, args...) = actime(x, par:par; vtype=vtype, args...)

actime(c::MCChain, pars::Ranges=1:size(c.samples, 2); vtype::Symbol=:imse, args...) =
  actime(c.samples, pars; vtype=vtype, args...)

actime(c::MCChain, par::Real; vtype::Symbol=:imse, args...) = actime(c, par:par; vtype=vtype, args...)
