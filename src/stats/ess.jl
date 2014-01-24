export ess, actime

# Effective sample size (ESS)
actypes = (:bm, :imse, :ipse)

function ess(c::MCMCChain, pars::Ranges=1:size(c.samples, 2); vtype::Symbol=:imse, args...)
  @assert in(vtype, actypes) "Unknown ESS type $vtype"

  return size(c.samples, 1)*var(c, pars; vtype=:iid)./var(c, pars; vtype=vtype, args...)
end

ess(c::MCMCChain, par::Real; vtype::Symbol=:imse, args...) = ess(c, par:par; vtype=vtype, args...)

# Integrated autocorrelation time
function actime(c::MCMCChain, pars::Ranges=1:size(c.samples, 2); vtype::Symbol=:imse, args...)
  @assert in(vtype, actypes) "Unknown integrated autocorrelation time type $vtype"

  return var(c, pars; vtype=vtype, args...)./var(c, pars; vtype=:iid)
end

actime(c::MCMCChain, par::Real; vtype::Symbol=:imse, args...) = actime(c, par:par; vtype=vtype, args...)
