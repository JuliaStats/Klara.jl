using Lora

plogtarget(p::Int, v::Vector) = log(v[1]^p/factorial(p))

p = BasicDiscUnvParameter(:p, logtarget=plogtarget, nkeys=2)

model = likelihood_model([Constant(:λ), p], isindexed=false)

function prandproposal(i::Int)
  s = (i == 0) ? [0, 1] : [i-1, i+1]
  rand(s)
end

sampler = MH(prandproposal)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:λ=>6, :p=>2)

job = BasicMCJob(model, sampler, mcrange, v0)

run(job)

chain = output(job)

mean(chain)

acceptance(chain, diagnostics=false)
