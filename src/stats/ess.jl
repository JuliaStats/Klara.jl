# Effective sample size (ESS)
function ess(mcmc::DataFrame; vtype::ASCIIString="imse")
  if vtype == "imse"
    return var(mcmc)/var_imse(mcmc, vtype=vtype)
  end
end

# Integrated autocorrelation time
function actime(mcmc::DataFrame; vtype::ASCIIString="imse")
  if vtype == "imse"
    return size(mcmc, 1)/ess(mcmcm, vtype=vtype)
  end
end
