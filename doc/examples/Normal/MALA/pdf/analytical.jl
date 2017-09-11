using Distributions, Klara

distribution = Normal()

p = BasicContUnvParameter(
  :p,
  logtarget=x::Float64 -> logpdf(distribution, x),
  gradlogtarget=x::Float64 -> gradlogpdf(distribution, x)
)

model = likelihood_model([p], isindexed=false)

mcsampler = MALA(0.95)

tuner = VanillaMCTuner()

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>3.11)

outopts = Dict(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, mcsampler, mcrange, v0, outopts=outopts)

@time run(job)

chain = output(job)

mean(chain)

acceptance(chain)
