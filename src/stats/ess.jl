export ess, actime

# Effective sample size (ESS)
actypes = (:imse, :ipse)

function ess(c::MCMCChain; vtype::Symbol=:imse)
  assert(in(vtype, actypes), "Unknown ESS type $vtype")

  return size(c.samples, 1)*var(c, vtype=:iid)./var(c, vtype=vtype)
end

# Integrated autocorrelation time
function actime(c::MCMCChain; vtype::Symbol=:imse)
  assert(in(vtype, actypes), "Unknown integrated autocorrelation time type $vtype")

  return var(c, vtype=vtype)./var(c, vtype=:iid)
end
