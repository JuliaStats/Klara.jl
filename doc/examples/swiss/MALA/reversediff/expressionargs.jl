using Lora

covariates, = dataset("swiss", "measurements")
ndata, npars = size(covariates)

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1)

outcome, = dataset("swiss", "status")
outcome = vec(outcome)

ploglikelihood =
  quote
    Xp = v[2]*p
    dot(Xp, v[3])-sum(log(1+exp(Xp)))
  end

plogprior = :(-0.5*(dot(p, p)/v[1]+length(p)*log(2*pi*v[1])))

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

init = Any[(:p, v0[:p]), (:v, Any[v0[:λ], v0[:X], v0[:y], v0[:p]])]

p = BasicContMuvParameter(:p, loglikelihood=ploglikelihood, logprior=plogprior, nkeys=4, autodiff=:reverse, init=init)

model = likelihood_model([Hyperparameter(:λ), Data(:X), Data(:y), p], isindexed=false)

sampler = MALA(0.1)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, outopts=outopts)

run(job)

chain = output(job)

mean(chain)

acceptance(chain)
