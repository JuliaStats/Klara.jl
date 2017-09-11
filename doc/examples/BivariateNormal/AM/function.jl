using Klara

# Precision matrix
C = Hyperparameter(:C)

p = BasicContMuvParameter(:p, logtarget=(p::Vector{Float64}, v::Vector) -> -dot(p, v[1]*p), nkeys=2)

model = GenericModel([C, p], isindexed=false)

mcsampler = AM(1., 2)

tuner = VanillaMCTuner()

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:C=>inv([1. 0.8; 0.8 1.]), :p=>Float64[1.25, 3.11])

outopts = Dict(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, mcsampler, mcrange, v0, outopts=outopts)

@time run(job)

chain = output(job)

mean(chain)

acceptance(chain)
