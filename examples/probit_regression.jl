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
priorvar = priorstd^2
distribution = MvNormal(zeros(size(X, 2)), priorvar*eye(npars))
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
    -(1-y).*exp(-(XPars.^2+log(2*pi))/2-logcdf(normal, -XPars)))-pars/priorvar)
end

# Define metric tensor
function tensor(pars::Vector{Float64})    
  XPars = X*pars
  normal = Normal()
  vector = exp(-XPars.^2-logcdf(normal, XPars)-logcdf(normal, -XPars)-log(2*pi))
  return ((X'.*repmat(vector', npars, 1))*X+(eye(npars)/priorvar))
end

# Define derivatives of metric tensor
function deriv_tensor(pars::Vector{Float64})
  output = Array(Float64, npars, npars, npars)
  
  XPars = X*pars
  normal = Normal()
  vector01 = exp(-XPars.^2-2*logcdf(normal, XPars)-logcdf(normal, -XPars)-log(2*pi))

  for i = 1:npars
    vector02 = (vector01.*(exp(-(XPars.^2+log(2*pi))/2-logcdf(normal, -XPars))
      -2*(pdf(normal, XPars)+XPars.*cdf(normal, XPars))).*X[:, i])
    
    output[:, :, i] = (X'.*repmat(vector02', npars, 1))*X
  end
  
  return output
end

mcmodel = model(log_posterior, grad=grad_log_posterior, tensor=tensor, dtensor=deriv_tensor, init=randprior())

mcchain01 = run(mcmodel * RWM(0.5) * SerialMC(1001:10000))

acceptance(mcchain01)

mcchain02 = run(mcmodel * HMC(0.001) * SerialMC(1001:10000))

acceptance(mcchain02)

mcchain03 = run(mcmodel * RMHMC(0.5, EmpMCTuner(0.8, verbose=true)) * SerialMC(5001:10000))

acceptance(mcchain03)
