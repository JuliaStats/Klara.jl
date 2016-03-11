using Lora

data, header = dataset("swiss")

covariates = data[:, 1:end-1]
ndata, npars = size(covariates)

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1)

outcome = data[:, end]

function ploglikelihood(p::Vector, v::Vector)
  Xp = v[2]*p
  dot(Xp, v[3])-sum(log(1+exp(Xp)))
end

plogprior(p::Vector, v::Vector) = -0.5*(dot(p, p)/v[1]+length(p)*log(2*pi*v[1]))

λ = Hyperparameter(:λ)

X = Data(:X)

y = Data(:y)

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  nkeys=4,
  autodiff=:forward,
  order=2
)

model = likelihood_model([λ, X, y, p], isindexed=false)

sampler = SMMALA(0.02)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, outopts=outopts)

run(job)

chain = output(job)

mean(chain)

acceptance(chain)
