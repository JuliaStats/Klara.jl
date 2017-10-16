using Klara

p = BasicContUnvParameter(:p, logtarget=p::Float64 -> -abs2(p))

model = likelihood_model([p], isindexed=false)

mcsampler = UnvAMWG()

tuner = RobertsRosenthalMCTuner()

mcrange = BasicMCRange(nsteps=100000, burnin=10000)

v0 = Dict(:p=>randn())

outopts = Dict(:monitor=>[:value, :logtarget], :diagnostics=>[:accept, :logσ])

job = BasicMCJob(model, mcsampler, mcrange, v0, tuner=tuner, outopts=outopts)

@time run(job)

chain = output(job)

mean(chain)
