### Functions for calculating zero variance (ZV) MC estimators, see  Mira A, Solgi R, Imparato D. Zero Variance Markov
### Chain Monte Carlo for Bayesian Estimators. Statistics and Computing, 2013, 23 (5), pp 653-662

### Functions for calculating ZV-MC estimators using linear polynomial

function linearzv(chain::Matrix{Float64}, grad::Matrix{Float64})
  npars = size(chain, 2)

  covAll = Array(Float64, npars+1, npars+1, npars)
  precision = Array(Float64, npars, npars, npars)
  sigma = Array(Float64, npars, npars)
  a = Array(Float64, npars, npars)

  z = -grad/2

  for i = 1:npars
     covAll[:, :, i] = cov([z chain[:, i]])
     precision[:, :, i] = inv(covAll[1:npars, 1:npars, i])
     sigma[:, i] = covAll[1:npars, npars+1, i]
     a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  chain = chain+z*a
  
  return chain, a
end

linearzv(c::MCChain) = linearzv(c.samples, c.gradlogtargets)

### Functions for calculating ZV-MC estimators using quadratic polynomial

function quadraticzv(chain::Matrix{Float64}, grad::Matrix{Float64})
  nsamples, npars = size(chain)
  k = convert(Int, npars*(npars+3)/2)
  l = 2*npars+1
  
  zQuadratic = Array(Float64, nsamples, k)  
  covAll = Array(Float64, k+1, k+1, npars)
  precision = Array(Float64, k, k, npars)
  sigma = Array(Float64, k, npars)
  a = Array(Float64, k, npars)

  z = -grad/2

  zQuadratic[:, 1:npars] = z
  zQuadratic[:, (npars+1):(2*npars)] = 2*z.*chain.-1
  for i = 1:(npars-1)
    for j = (i+1):npars
      zQuadratic[:, l] = chain[:, i].*z[:, j]+chain[:, j].*z[:, i]
      l += 1
    end
  end

  for i = 1:npars
    covAll[:, :, i] = cov([zQuadratic chain[:, i]]);
    precision[:, :, i] = inv(covAll[1:k, 1:k, i])
    sigma[:, i] = covAll[1:k, k+1, i]
    a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  zvChain = chain+zQuadratic*a;

  return zvChain, a
end

quadraticzv(c::MCChain) = quadraticzv(c.samples, c.gradlogtargets)
