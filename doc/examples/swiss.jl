using Lora

data, header = dataset("swiss")

covariates = data[:, 1:end-1]
ndata, npars = size(covariates)

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1)

outcome = data[:, end]

function ploglikelihood(p::Vector{Float64}, v::Vector)
  Xp = v[2]*p
  (Xp'*v[3]-sum(log(1+exp(Xp))))[1]
end

plogprior(p::Vector{Float64}, v::Vector) = -0.5*(dot(p, p)/v[1]+length(p)*log(2*pi*v[1]))

λ = Hyperparameter(:λ)

X = Data(:X)
y = Data(:y)

p = BasicContMuvParameter(:p, loglikelihood=ploglikelihood, logprior=plogprior, nkeys=4)

model = likelihood_model([λ, X, y, p], isindexed=false)

sampler = MH(ones(npars))

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

job = BasicMCJob(model, sampler, mcrange, v0)

run(job)

chain = output(job)

[mean(chain.value[i, :]) for i in 1:4]
