using Klara

# Precision matrix
C = Hyperparameter(:C)

p = BasicContMuvParameter(:p, logtarget=(p::Vector{Float64}, v::Vector) -> -dot(p, v[1]*p), nkeys=2)

model = GenericModel([C, p], isindexed=false)

mcsampler = MuvAMWG([1., 2.])

tuner = RobertsRosenthalMCTuner(verbose=true)

mcrange = BasicMCRange(nsteps=100000, burnin=10000)

v0 = Dict(:C=>inv([1. 0.8; 0.8 1.]), :p=>Float64[1.25, 3.11])

outopts = Dict(:monitor=>[:value, :logtarget], :diagnostics=>[:accept, :logσ])

job = BasicMCJob(model, mcsampler, mcrange, v0, tuner=tuner, outopts=outopts)

@time run(job)

chain = output(job)

mean(chain)

cor(chain.value[1, :], chain.value[2, :])
