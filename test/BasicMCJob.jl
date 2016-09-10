using Base.Test
using Klara

# println("    Testing BasicMCJob constructors...")

# Example 1: multivariate parameter

p = BasicContMuvParameter(
  :p, 1, signature=:low, logtarget=(state, states) -> state.logtarget = -dot(state.value, state.value)
)
model = likelihood_model([p])

sampler = MH([1., 1.])

tuner = VanillaMCTuner()
# tuner = VanillaMCTuner(verbose=true)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

vstate = [BasicContMuvParameterState([1.25, 3.11], [:value, :logtarget], [:accept])]

outopts = Dict(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

job = BasicMCJob(
  model,
  1,
  sampler,
  tuner,
  mcrange,
  vstate,
  outopts,
  true,
  true,
  false,
  false
)

@time run(job)

chain = output(job)

mean(chain)

acceptance(chain)

# Example 2: univariate parameter

# using Klara

p = BasicContUnvParameter(:p, 1, signature=:low, logtarget=(state, states) -> state.logtarget = -abs2(state.value))
model = likelihood_model([p])

sampler = MH()

tuner = VanillaMCTuner()

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

vstate = [BasicContUnvParameterState(5.1)]

outopts = Dict(:monitor=>[:value, :logtarget])

job = BasicMCJob(
  model,
  1,
  sampler,
  tuner,
  mcrange,
  vstate,
  outopts,
  true,
  true,
  false,
  false
)

@time run(job)

chain = output(job)

mean(chain)

acceptance(chain, diagnostics=false)
