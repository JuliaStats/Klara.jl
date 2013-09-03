export BayesLogitModel, BayesProbitModel, loglik, logposterior

abstract BayesGLMModel <: Model

type BayesLogitModel <: BayesGLMModel
  X::Array{Float64, 2} # Design matrix
  y::Vector{Float64} # Response variable
  ndata::Int # Number of data samples
  npars::Int # Number of parameters (X is an ndata*npars matrix)
  
  logprior::Function
  
  function BayesLogitModel(X::Array{Float64, 2}, y::Vector{Float64}, ndata::Int, npars::Int, logprior::Function)
    assert(ndata > 0, "Number of data samples should be > 0")
    assert(npars > 0, "Number of model parameters should be > 0")
    assert(size(X) == (ndata, npars), "Design matrix dimensions and (ndata, npars) not consistent")
    assert(length(y) == ndata, "Response variable vector length and ndata not consistent")

    assert(!isgeneric(logprior) | length(methods(logprior, (Vector{Float64},))) == 1, "logprior cannot be called with Vector{Float64}")
    
    new(X, y, ndata, npars, logprior)
  end
end

BayesLogitModel(X::Array{Float64, 2}, y::Vector{Float64}, logprior::Function) = BayesLogitModel(X, y, size(X, 1), size(X, 2), logprior)

function loglik(pars::Vector{Float64}, model::BayesLogitModel)
  XPars = model.X*pars
  return (XPars'*model.y-sum(log(1+exp(XPars))))[1]
end

logposterior(pars::Vector{Float64}, model::BayesLogitModel) = loglik(pars, model)+model.logprior(pars)

type BayesProbitModel <: BayesGLMModel
  X::Array{Float64, 2} # Design matrix
  y::Vector{Float64} # Response variable
  ndata::Int # Number of data samples
  npars::Int # Number of parameters (X is an ndata*npars matrix)
  
  logprior::Function
  
  function BayesProbitModel(X::Array{Float64, 2}, y::Vector{Float64}, ndata::Int, npars::Int, logprior::Function)
    assert(ndata > 0, "Number of data samples should be > 0")
    assert(npars > 0, "Number of model parameters should be > 0")
    assert(size(X) == (ndata, npars), "Design matrix dimensions and (ndata, npars) not consistent")
    assert(length(y) == ndata, "Response variable vector length and ndata not consistent")

    assert(!isgeneric(logprior) | length(methods(logprior, (Vector{Float64},))) == 1, "logprior cannot be called with Vector{Float64}")
    
    new(X, y, ndata, npars, logprior)
  end
end

BayesProbitModel(X::Array{Float64, 2}, y::Vector{Float64}, logprior::Function) = BayesProbitModel(X, y, size(X, 1), size(X, 2), logprior)

function loglik(pars::Vector{Float64}, model::BayesProbitModel)  
  XPars = model.X*pars
  normal = Normal()
  return (dot(logcdf(normal, XPars), model.y)+dot(logcdf(normal, -XPars), 1-model.y))
end

logposterior(pars::Vector{Float64}, model::BayesProbitModel) = loglik(pars, model)+model.logprior(pars)

function grad(pars::Vector{Float64}, model::BayesProbitModel)
  return (model.X'*(model.y-1./(1+exp(-model.X*pars)))-pars/data["priorVar"])
end