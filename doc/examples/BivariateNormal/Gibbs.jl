# https://theclevermachine.wordpress.com/2012/11/05/mcmc-the-gibbs-mcsampler/

using Distributions
using Klara

ρ = Hyperparameter(:ρ)

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

model = GenericModel([ρ, p1, p2], isindexed=false)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:ρ=>0.8, :p1=>5.1, :p2=>2.3)

job = BasicGibbsJob(model, Dict(), mcrange, v0)

@time run(job)

output(job)

chains = Dict(job)

[mean(chains[k]) for k in keys(chains)]

cor(chains[:p1].value, chains[:p2].value)
