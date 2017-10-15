using Klara

covariates, = dataset("swiss", "measurements")
ndata, npars = size(covariates)

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1)

outcome, = dataset("swiss", "status")
outcome = vec(outcome)

function ploglikelihood(p::Vector{Float64}, v::Vector)
  Xp = v[2]*p
  dot(Xp, v[3])-sum(log.(1+exp.(Xp)))
end

plogprior(p::Vector{Float64}, v::Vector) = -0.5*(dot(p, p)/v[1]+length(p)*log(2*pi*v[1]))

p = BasicContMuvParameter(:p, loglikelihood=ploglikelihood, logprior=plogprior, nkeys=4)

model = likelihood_model([Hyperparameter(:λ), Data(:X), Data(:y), p], isindexed=false)

mcsampler = MuvAMWG([2.5, 1., 3., 2.5])

tuner = RobertsRosenthalMCTuner(verbose=true)

mcrange = BasicMCRange(nsteps=100000, burnin=10000)

v0 = Dict(:λ=>100, :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

outopts = Dict(:monitor=>[:value, :logtarget], :diagnostics=>[:accept, :logσ])

job = BasicMCJob(model, mcsampler, mcrange, v0, tuner=tuner, outopts=outopts)

run(job)

chain = output(job)

mean(chain)
