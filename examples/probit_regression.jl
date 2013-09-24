# Bayesian probit model with a Normal prior N(0, priorstd^2*I). The log-posterior and its gradient, as well as the
# metric tensor and its partial derivatives are passed to the model as functions

using Distributions, MCMC

# Create design matrix X and response variable y from vaso data array
vaso = readdlm("vaso.txt", ' ')
covariates, y = vaso[:, 1:end-1], vaso[:, end]
nsamples, npars = size(covariates)
covariates = (broadcast(-, covariates, mean(covariates, 1))./repmat(std(covariates, 1), nsamples, 1))
polynomialOrder = 1
X = ones(nsamples, npars*polynomialOrder+1)
for i = 1:polynomialOrder
  X[:, ((i-1)*npars+2):(i*npars+1)] = covariates.^i
end
npars += 1

# Define log-prior
priorstd = 10.
distribution = MvNormal(size(X, 2), priorstd)
logprior(pars::Vector{Float64}) = logpdf(distribution, pars)
randprior() = rand(distribution)

# Define log-likelihood
function loglik(pars::Vector{Float64})
  XPars = X*pars
  normal = Normal()
  return (dot(logcdf(normal, XPars), y)+dot(logcdf(normal, -XPars), 1-y))
end

# Define log-posterior
log_posterior(pars::Vector{Float64}) = logprior(pars)+loglik(pars)

# Define gradient of log-posterior
function grad_log_posterior(pars::Vector{Float64})    
  XPars = X*pars
  normal = Normal()
  return (X'*(y.*exp(-(XPars.^2+log(2*pi))/2-logcdf(normal, XPars))
    -(1-y).*exp(-(XPars.^2+log(2*pi))/2-logcdf(normal, XPars)))-pars/(priorstd^2))
end

mcmodel = mcmodel = model(log_posterior, grad=grad_log_posterior, init=randprior())

mcchain01 = mcmodel * RWM(0.5) * (1001:10000)

acceptance(mcchain01)

mcchain02 = mcmodel * HMC(0.001) * (1001:10000)

acceptance(mcchain02)
