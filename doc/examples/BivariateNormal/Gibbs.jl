using Distributions
using Lora

ρ = Hyperparameter(:ρ, 1)

p1 = BasicContUnvParameter(
  :p1,
  setpdf=(p::Float64, v::Vector) -> Normal(v[1]*v[3], sqrt(1-abs2(v[1]))),
  nkeys=3
)

p2 = BasicContUnvParameter(
  :p2,
  setpdf=(p::Float64, v::Vector) -> Normal(v[1]*v[2], sqrt(1-abs2(v[1]))),
  nkeys=3
)

model = GenericModel([ρ, p1, p2], [ρ p1; ρ p2; p1 p2], isindexed=false)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

vstate = [BasicUnvVariableState(0.8), BasicContUnvParameterState(5.1), BasicContUnvParameterState(2.3)]

outopts = [Dict(:monitor=>[:value]), Dict(:monitor=>[:value])]

job = GibbsJob(
  model,
  [2, 3],
  [nothing, nothing],
  mcrange,
  vstate,
  outopts,
  true,
  true,
  false,
  false
)

@time run(job)

output(job)

chains = Dict(job)

[mean(chains[k]) for k in keys(chains)]

cor(chains[:p1].value, chains[:p2].value)
