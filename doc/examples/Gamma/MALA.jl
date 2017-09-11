# Apply a log-transform on a Gamma-distributed variable to sample from it using a normal proposal
# Use forward-mode automatic differentiation to compute the gradient of the log-target

using Klara

plogtarget(p::Real, v::Vector) = (v[1]-1)*p-exp(p)/v[2]+p

p = BasicContUnvParameter(:p, logtarget=plogtarget, nkeys=3, diffopts=DiffOptions(mode=:forward))

model = likelihood_model([Constant(:k), Constant(:θ), p], isindexed=false)

mcsampler = MALA()

mcrange = BasicMCRange(nsteps=100000, burnin=10000)

v0 = Dict(:k=>2., :θ=>1., :p=>log(10.))

job = BasicMCJob(model, mcsampler, mcrange, v0)

run(job)

chain = output(job)

mean(exp.(chain.value))

acceptance(chain, diagnostics=false)
