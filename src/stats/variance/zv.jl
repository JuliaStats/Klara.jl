### Zero variance (ZV) Monte Carlo estimators
### Reference:
### Mira A, Solgi R and Imparato D
### Zero Variance Markov Chain Monte Carlo for Bayesian Estimators
### Statistics and Computing, 2013, 23 (5), pp 653-662

## Zero variance (ZV) Monte Carlo estimators linear polynomials

function lzv{N<:Real}(chain::Vector{N}, grad::Vector{N})
  z = -0.5*grad
  augmentedcov = cov([z chain])
  a = -augmentedcov[1, 2]/augmentedcov[1, 1]
  return chain+z*a, a
end

function lzv{N<:Real}(chain::Matrix{N}, grad::Matrix{N})
  npars = size(chain, 2)

  augmentedcov = Array(N, npars+1, npars+1, npars)
  precision = Array(N, npars, npars, npars)
  sigma = Array(N, npars, npars)
  a = Array(N, npars, npars)

  z = -0.5*grad

  for i = 1:npars
     augmentedcov[:, :, i] = cov([z chain[:, i]])
     precision[:, :, i] = inv(augmentedcov[1:npars, 1:npars, i])
     sigma[:, i] = augmentedcov[1:npars, npars+1, i]
     a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  return chain+z*a, a
end

lzv(s::ParameterNState{Continuous, Univariate}) = lzv(s.value, s.gradlogtarget)

lzv(s::ParameterNState{Continuous, Multivariate}) = lzv(transpose(s.value), transpose(s.gradlogtarget))
