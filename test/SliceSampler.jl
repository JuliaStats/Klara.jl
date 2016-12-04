using Klara

function plogtarget(x::Vector{Float64})
  s2 = exp(x[1])
  nx = length(x)-1
  -0.5*((x[1]-0.0)^2/9.0+dot(x[2:end], x[2:end])/s2+nx*log(s2))
end

p = BasicContMuvParameter(:p, logtarget=plogtarget)

model = likelihood_model(p, false)

sampler = SliceSampler(1., 5)

mcrange = BasicMCRange(nsteps=100000, burnin=10000)

v0 = Dict(:p=>[10.; zeros(4)])

job = BasicMCJob(model, sampler, mcrange, v0)

run(job)

chain = output(job)

mean(chain)
