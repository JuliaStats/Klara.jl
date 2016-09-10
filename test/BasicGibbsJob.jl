using Base.Test
using Distributions
using Klara

ρ = Hyperparameter(:ρ, 1)

p1 = BasicContUnvParameter(
  :p1,
  2,
  signature=:low,
  setpdf=(state, states) -> Normal(states[1].value*states[3].value, sqrt(1-abs2(states[1].value)))
)

p2 = BasicContUnvParameter(
  :p2,
  3,
  signature=:low,
  setpdf=(state, states) -> Normal(states[1].value*states[2].value, sqrt(1-abs2(states[1].value)))
)

model = GenericModel([ρ, p1, p2], [ρ p1; ρ p2; p1 p2])

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

vstate = [BasicUnvVariableState(0.8), BasicContUnvParameterState(5.1), BasicContUnvParameterState(2.3)]

outopts = [Dict(:monitor=>[:value]), Dict(:monitor=>[:value])]

job = BasicGibbsJob(
  model,
  [2, 3],
  [nothing, nothing],
  mcrange,
  vstate,
  outopts,
  true,
  false,
  false
)

@time run(job)

# output(job)

chains = Dict(job)

[mean(chains[k]) for k in keys(chains)]

cor(chains[:p1].value, chains[:p2].value)
