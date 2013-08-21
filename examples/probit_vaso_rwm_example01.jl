using Distributions, MCMC

include("vaso.jl");

priorstd = 10.;
distribution = MvNormal(size(X, 2), priorstd);
logprior(pars::Vector{Float64}) = logpdf(distribution, pars);
probitmodel = BayesProbitModel(X, y, logprior);

logtarget(pars::Vector{Float64}) = logposterior(pars, probitmodel);
randprior() = rand(distribution);

mcmcmodel = MCMCLikModel(logtarget, probitmodel.npars, randprior());

mcmcchain = mcmcmodel * RWM(0.1) * (5001:55000)

mcmcchain = mcmcmodel * RWM(0.1) * (5001:5:55000)
