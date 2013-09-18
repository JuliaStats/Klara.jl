export ess, actime

# Effective sample size (ESS)
function ess(c::MCMCChain; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown ESS type $vtype")

  if vtype == :imse
    return size(c.samples, 1)*var(c, vtype=:iid)./var(c, vtype=vtype)
  end
end

# Integrated autocorrelation time
function actime(c::MCMCChain; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown integrated autocorrelation time type $vtype")

  if vtype == :imse
    return var(c, vtype=vtype)./var(c, vtype=:iid)
  end
end
