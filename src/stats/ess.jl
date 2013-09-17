export ess, actime

# Effective sample size (ESS)
function ess(mcmc::DataFrame; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown ESS type $vtype")

  if vtype == :imse
    return var(mcmc)/var_imse(mcmc, vtype=vtype)
  end
end

# Integrated autocorrelation time
function actime(mcmc::DataFrame; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown integrated autocorrelation time type $vtype")

  if vtype == :imse
    return size(mcmc, 1)/ess(mcmcm, vtype=vtype)
  end
end
