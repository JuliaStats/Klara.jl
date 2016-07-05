using Lora

plogtarget(p::Int, v::Vector) = p*log(v[1])-log(factorial(p))

p = BasicDiscUnvParameter(:p, logtarget=plogtarget, nkeys=2)

model = likelihood_model([Constant(:λ), p], isindexed=false)

psetproposal(i::Int) = (i == 0) ? Binary(0, 1) : Binary(i-1, i+1)

sampler = MH(psetproposal, symmetric=false)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:λ=>6, :p=>2)

job = BasicMCJob(model, sampler, mcrange, v0)

run(job)

chain = output(job)

mean(chain)

acceptance(chain, diagnostics=false)
