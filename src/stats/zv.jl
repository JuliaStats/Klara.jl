export linearZv, quadraticZv

# Function for calculating ZV-c.samples estimators using linear polynomial
function linearZv(c::MCMCChain)
  npars = size(c.samples, 2)

  covAll = Array(Float64, npars+1, npars+1, npars)
  precision = Array(Float64, npars, npars, npars)
  sigma = Array(Float64, npars, npars)
  a = Array(Float64, npars, npars)

  zvChain, z = matrix(c.samples), matrix(-c.gradients/2)

  for i = 1:npars
     covAll[:, :, i] = cov([z zvChain[:, i]])
     precision[:, :, i] = inv(covAll[1:npars, 1:npars, i])
     sigma[:, i] = covAll[1:npars, npars+1, i]
     a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  zvChain = zvChain+z*a
  
  return zvChain, a
end

# Function for calculating ZV-c.samples estimators using quadratic polynomial
function quadraticZv(c::MCMCChain)
  nsamples, npars = size(c.samples)
  k = convert(Int, npars*(npars+3)/2)
  l = 2*npars+1
  
  zQuadratic = Array(Float64, nsamples, k)  
  covAll = Array(Float64, k+1, k+1, npars)
  precision = Array(Float64, k, k, npars)
  sigma = Array(Float64, k, npars)
  a = Array(Float64, k, npars)

  zvChain, z = matrix(c.samples), matrix(-c.gradients/2)

  zQuadratic[:, 1:npars] = z
  zQuadratic[:, (npars+1):(2*npars)] = 2*z.*zvChain-1
  for i = 1:(npars-1)
    for j = (i+1):npars
      zQuadratic[:, l] = zvChain[:, i].*z[:, j]+zvChain[:, j].*z[:, i]
      l += 1
    end
  end

  for i = 1:npars
    covAll[:, :, i] = cov([zQuadratic zvChain[:, i]]);
    precision[:, :, i] =
      inv(covAll[1:k, 1:k, i])
    sigma[:, i] = covAll[1:k, k+1, i]
    a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  zvChain = zvChain+zQuadratic*a;

  return zvChain, a
end
