using Klara

covariates, = dataset("swiss", "measurements")
ndata, npars = size(covariates)

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1)

outcome, = dataset("swiss", "status")
outcome = vec(outcome)

function ploglikelihood(p, v)
  Xp = v[2]*p
  dot(Xp, v[3])-sum(log.(1+exp.(Xp)))
end

plogprior(p, v) = -0.5*(dot(p, p)/v[1]+length(p)*log(2*pi*v[1]))

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  nkeys=4,
  diffopts=DiffOptions(mode=:reverse)
)

model = likelihood_model([Hyperparameter(:λ), Data(:X), Data(:y), p], isindexed=false)

mcsampler = HMC(0.35)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, mcsampler, mcrange, v0, outopts=outopts)

run(job)

chain = output(job)

mean(chain)

acceptance(chain)
