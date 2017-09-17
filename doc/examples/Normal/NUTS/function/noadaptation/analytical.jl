using Klara

p = BasicContUnvParameter(:p, logtarget=p::Float64 -> -abs2(p), gradlogtarget=p::Float64 -> -2*p)

model = likelihood_model([p], isindexed=false)

mcsampler = NUTS(0.4, maxndoublings=7)

tuner = VanillaMCTuner()

mcrange = BasicMCRange(nsteps=5000, burnin=1000)

v0 = Dict(:p=>3.11)

outopts = Dict(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept, :ndoublings])

job = BasicMCJob(model, mcsampler, mcrange, v0, tuner=tuner, outopts=outopts)

@time run(job)

chain = output(job)

mean(chain)

diags = diagnostics(chain)
