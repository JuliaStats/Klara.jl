### Effective sample size (ESS)

actypes = (:bm, :imse, :ipse)

function ess(x::Vector{Float64}; vtype::Symbol=:imse, args...)
  @assert in(vtype, actypes) "Unknown ESS type $vtype"

  return length(x)*mcvar(x; vtype=:iid)/mcvar(x; vtype=vtype, args...)
end

ess(x::Matrix{Float64}, pars::Range=1:size(x, 2); vtype::Symbol=:imse, args...) =
  Float64[ess(x[:, pars[i]]; vtype=vtype, args...) for i = 1:length(pars)]

ess(x::Matrix{Float64}, par::Real; vtype::Symbol=:imse, args...) = ess(x, par:par; vtype=vtype, args...)

ess(c::MCChain, pars::Range=1:size(c.samples, 2); vtype::Symbol=:imse, args...) =
  ess(c.samples, pars; vtype=vtype, args...)

ess(c::MCChain, par::Real; vtype::Symbol=:imse, args...) = ess(c, par:par; vtype=vtype, args...)
