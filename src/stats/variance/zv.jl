### Zero variance (ZV) Monte Carlo estimators
### Reference:
### Mira A, Solgi R and Imparato D
### Zero Variance Markov Chain Monte Carlo for Bayesian Estimators
### Statistics and Computing, 2013, 23 (5), pp 653-662

## Zero variance (ZV) Monte Carlo estimators linear polynomials

function lzv(chain::Vector{N}, grad::Vector{N}) where N<:Real
  z = -0.5*grad
  augmentedcov = cov([z chain])
  a = -augmentedcov[1, 2]/augmentedcov[1, 1]
  return chain+z*a, a
end

function lzv(chain::Matrix{N}, grad::Matrix{N}) where N<:Real
  npars = size(chain, 2)

  augmentedcov = Array(N, npars+1, npars+1, npars)
  precision = Array(N, npars, npars, npars)
  sigma = Array(N, npars, npars)
  a = Array(N, npars, npars)

  z = -0.5*grad

  for i in 1:npars
     augmentedcov[:, :, i] = cov([z chain[:, i]])
     precision[:, :, i] = inv(augmentedcov[1:npars, 1:npars, i])
     sigma[:, i] = augmentedcov[1:npars, npars+1, i]
     a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  return chain+z*a, a
end

lzv(s::ParameterNState{Continuous, Univariate}) = lzv(s.value, s.gradlogtarget)

lzv(s::ParameterNState{Continuous, Multivariate}) = lzv(transpose(s.value), transpose(s.gradlogtarget))

## Zero variance (ZV) Monte Carlo estimators quadratic polynomials

function qzv(chain::Vector{N}, grad::Vector{N}) where N<:Real
  z = -0.5*grad
  qz = [z 2*z.*chain-1]
  augmentedcov = cov([qz chain])
  a = -inv(augmentedcov[1:2, 1:2])*augmentedcov[1:2, 3]
  return chain+qz*a, a
end

function qzv(chain::Matrix{N}, grad::Matrix{N}) where N<:Real
  nsamples, npars = size(chain)
  k = convert(Integer, npars*(npars+3)/2)
  l = 2*npars+1

  qz = Array(N, nsamples, k)
  augmentedcov = Array(N, k+1, k+1, npars)
  precision = Array(N, k, k, npars)
  sigma = Array(N, k, npars)
  a = Array(N, k, npars)

  z = -grad/2

  qz[:, 1:npars] = z
  qz[:, (npars+1):(2*npars)] = 2*z.*chain-1
  for i in 1:(npars-1)
    for j = (i+1):npars
      qz[:, l] = chain[:, i].*z[:, j]+chain[:, j].*z[:, i]
      l += 1
    end
  end

  for i in 1:npars
    augmentedcov[:, :, i] = cov([qz chain[:, i]])
    precision[:, :, i] = inv(augmentedcov[1:k, 1:k, i])
    sigma[:, i] = augmentedcov[1:k, k+1, i]
    a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  return chain+qz*a, a
end

qzv(s::ParameterNState{Continuous, Univariate}) = qzv(s.value, s.gradlogtarget)

qzv(s::ParameterNState{Continuous, Multivariate}) = qzv(transpose(s.value), transpose(s.gradlogtarget))
