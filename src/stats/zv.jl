export linearZv, quadraticZv

# Function for calculating ZV-MCMC estimators using linear polynomial
function linearZv(mcmc::DataFrame, grad::DataFrame)
  nPars = size(mcmc, 2)

  covAll = Array(Float64, nPars+1, nPars+1, nPars)
  precision = Array(Float64, nPars, nPars, nPars)
  sigma = Array(Float64, nPars, nPars)
  a = Array(Float64, nPars, nPars)

  mcmc, z = matrix(mcmc), matrix(-grad/2)

  for i = 1:nPars
     covAll[:, :, i] = cov([z mcmc[:, i]])
     precision[:, :, i] = inv(covAll[1:nPars, 1:nPars, i])
     sigma[:, i] = covAll[1:nPars, nPars+1, i]
     a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  zvMcmc = mcmc+z*a
  
  return zvMcmc, a
end

# Function for calculating ZV-MCMC estimators using quadratic polynomial
function quadraticZv(mcmc::DataFrame, grad::DataFrame)
  nData, nPars = size(mcmc)
  k = convert(Int, nPars*(nPars+3)/2)
  l = 2*nPars+1
  
  zQuadratic = Array(Float64, nData, k)  
  covAll = Array(Float64, k+1, k+1, nPars)
  precision = Array(Float64, k, k, nPars)
  sigma = Array(Float64, k, nPars)
  a = Array(Float64, k, nPars)

  mcmc, z = matrix(mcmc), matrix(-grad/2)

  zQuadratic[:, 1:nPars] = z
  zQuadratic[:, (nPars+1):(2*nPars)] = 2*z.*mcmc-1
  for i = 1:(nPars-1)
    for j = (i+1):nPars
      zQuadratic[:, l] = mcmc[:, i].*z[:, j]+mcmc[:, j].*z[:, i]
      l += 1
    end
  end

  for i = 1:nPars
    covAll[:, :, i] = cov([zQuadratic mcmc[:, i]]);
    precision[:, :, i] =
      inv(covAll[1:k, 1:k, i])
    sigma[:, i] = covAll[1:k, k+1, i]
    a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  zvMcmc = mcmc+zQuadratic*a;

  return zvMcmc, a
end
