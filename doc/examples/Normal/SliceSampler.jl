using Klara

p = BasicContUnvParameter(:p, logtarget=p::Float64 -> -abs2(p))

model = likelihood_model(p, false)

sampler = SliceSampler(1., 5)

mcrange = BasicMCRange(nsteps=100000, burnin=10000)

v0 = Dict(:p=>3.11)

job = BasicMCJob(model, sampler, mcrange, v0)

run(job)

chain = output(job)

mean(chain)
