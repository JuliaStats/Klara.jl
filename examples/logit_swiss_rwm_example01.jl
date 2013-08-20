using Distributions, MCMC

include("swiss.jl");

priorstd = 10.;
distribution = MvNormal(size(X, 2), priorstd);
logprior(pars::Vector{Float64}) = logpdf(distribution, pars);
logitmodel = BayesLogitModel(X, y, logprior);

logtarget(pars::Vector{Float64}) = logposterior(pars, logitmodel);
randprior() = rand(distribution);

mcmcmodel = MCMCLikModel(logtarget, logitmodel.npars, randprior());

mcmcchain = mcmcmodel * RWM(0.1) * (5001:55000);

#mean(mcmcchain.samples, 2)
