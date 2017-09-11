### https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/

using Distributions
using Klara

plogtarget(p::Float64, v::Vector) = (v[1]-1)*log(p)-p/v[2]

p = BasicContUnvParameter(:p, logtarget=plogtarget, nkeys=3)

model = likelihood_model([Constant(:k), Constant(:θ), p], isindexed=false)

psetproposal(x::Float64) = Truncated(Normal(x), 0, Inf)

mcsampler = MH(psetproposal, symmetric=false)

mcrange = BasicMCRange(nsteps=100000, burnin=10000)

v0 = Dict(:k=>2., :θ=>1., :p=>10.)

job = BasicMCJob(model, mcsampler, mcrange, v0)

run(job)

chain = output(job)

mean(chain)

acceptance(chain, diagnostics=false)
