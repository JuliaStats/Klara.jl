using Distributions, MCMC

include("swiss.jl");

priorstd = 10.;
distribution = MvNormal(size(X, 2), priorstd);
logprior(pars::Vector{Float64}) = logpdf(distribution, pars);
logitmodel = BayesLogitModel(X, y, logprior);

l(pars::Vector{Float64}) = logposterior(pars, logitmodel);
randprior() = rand(distribution);

function gradlogposterior(pars::Vector{Float64}, model::BayesLogitModel)
  return (model.X'*(model.y-1./(1+exp(-model.X*pars)))-pars/(priorstd^2))
end

grad(pars::Vector{Float64}) = gradlogposterior(pars, logitmodel);

function tensor(pars::Vector{Float64}, model::BayesLogitModel)
  p = 1./(1+exp(-model.X*pars))
  return ((model.X'.*repmat((p.*(1-p))', model.npars, 1))*model.X+(eye(model.npars)/priorstd^2))
end

t(pars::Vector{Float64}) = tensor(pars, logitmodel);

function derivTensor(pars::Vector{Float64}, model::BayesLogitModel)
  matrix = Array(Float64, model.ndata, model.npars)
  output = Array(Float64, model.npars, model.npars, model.npars)
  
  p = 1./(1+exp(-model.X*pars))
  
  for i = 1:model.npars
    for j =1:model.npars
      matrix[:, j] = (model.X)[:, j].*((p.*(1-p)).*((1-2*p).*(model.X)[:, i]))
    end
    
    output[:, :, i] = matrix'*model.X
  end
  
  return output
end

dt(pars::Vector{Float64}) = derivTensor(pars, logitmodel);

mcmcmodel = model(l, grad, t, dt, init=randprior());

mcmcchain01 = mcmcmodel * RMHMC(0.1) * (101:1000)
