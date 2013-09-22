export ess, actime

# Effective sample size (ESS)
actypes = (:bm, :imse, :ipse)

function ess(c::MCMCChain; vtype::Symbol=:imse, args...)
  assert(in(vtype, actypes), "Unknown ESS type $vtype")

  return nrow(c.samples)*var(c, vtype=:iid)./var(c, vtype=vtype, args...)
end

# Integrated autocorrelation time
function actime(c::MCMCChain; vtype::Symbol=:imse, args...)
  assert(in(vtype, actypes), "Unknown integrated autocorrelation time type $vtype")

  return var(c, vtype=vtype)./var(c, vtype=:iid, args...)
end
