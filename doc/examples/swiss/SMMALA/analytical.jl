using Klara

covariates, = dataset("swiss", "measurements")
ndata, npars = size(covariates)

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1)

outcome, = dataset("swiss", "status")
outcome = vec(outcome)

function ploglikelihood(p::Vector, v::Vector)
  Xp = v[2]*p
  dot(Xp, v[3])-sum(log.(1+exp.(Xp)))
end

plogprior(p::Vector, v::Vector) = -0.5*(dot(p, p)/v[1]+length(p)*log(2*pi*v[1]))

pgradlogtarget(p::Vector, v::Vector) = v[2]'*(v[3]-1./(1+exp.(-v[2]*p)))-p/v[1]

function ptensorlogtarget(p::Vector, v::Vector)
  r = 1./(1+exp.(-v[2]*p))
  (v[2]'.*repmat((r.*(1-r))', length(p), 1))*v[2]+(eye(length(p))/v[1])
end

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget,
  nkeys=4
)

model = likelihood_model([Hyperparameter(:λ), Data(:X), Data(:y), p], isindexed=false)

mcsampler = SMMALA(0.02)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, mcsampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=true), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

acceptance(chain)
