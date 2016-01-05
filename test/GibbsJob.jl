using Base.Test
using Distributions
using Lora

ρ = Hyperparameter(:ρ, 1)

p1 = BasicContUnvParameter(
  :p1,
  2,
  setpdf=(state, states) -> Normal(states[1].value*states[3].value, sqrt(1-abs2(states[1].value)))
)

p2 = BasicContUnvParameter(
  :p2,
  3,
  setpdf=(state, states) -> Normal(states[1].value*states[2].value, sqrt(1-abs2(states[1].value)))
)

model = GenericModel([ρ, p1, p2], [ρ p1; ρ p2; p1 p2])

mcrange = BasicMCRange(nsteps=100, burnin=10)

vstate = [
  BasicUnvVariableState(0.8),
  BasicContUnvParameterState(5.1, [:value]),
  BasicContUnvParameterState(2.3, [:value]),
]

outopts = Dict{Symbol,Any}[Dict(:monitor=>[:value]), Dict(:monitor=>[:value])]

GibbsJob(
  model,
  [2, 3],
  Union{BasicMCJob, Void}[nothing, nothing],
  mcrange,
  vstate,
  outopts,
  true,
  true,
  false
)
