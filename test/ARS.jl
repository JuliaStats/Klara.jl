using Base.Test
using Distributions
# using Gadfly
using Klara

# Target distribution shape, unscaled
g(x) = exp(-dot(x, x)/2)
log_g(x) = -dot(x, x)/2

# Candidate (starting) density, use Normal(0, 2)
g0(x) = 1/(sqrt(2*pi)*2)*exp(-dot(x, x)/8)
log_g0(x) = -dot(x, x)/8-log(2 * sqrt(2*pi))

# Compute M0 such that candidate dominates target
h0(x) = -g(x)/g0(x)

#res0 = optimize(h0, ones(L))
#M0 = -res0.f_minimum

M0 = 10.0
log_M0 = log(M0)

# Candidate (starting) Distribution, use Normal(0, 2)
g1(x) = exp(sum(logpdf(Normal(0, 2), x)))
log_g1(x) = sum(logpdf(Normal(0, 2), x))

# Compute M1 such that candidate dominates target
h1(x) = -g(x)/sum(g1(x))
#res1 = optimize(h1, ones(L))
#M1 = -res1.f_minimum
M1 = 10.0
log_M1 = log(M1)

model = likelihood_model(BasicContUnvParameter(:p, logtarget=log_g), false)

sampler = ARS(log_g0, proposalscale=log_M0)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>1.)


job = BasicMCJob(model, sampler, mcrange, v0)

run(job)

chain1 = output(job)

sampler = ARS(log_g1, proposalscale=log_M1)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>1.)


job = BasicMCJob(model, sampler, mcrange, v0)

run(job)

chain2= output(job)

println()
println("M: ", [M0 M1])
println()

println("Acceptance rate: ", acceptance(chain1, diagnostics=false))

println("Acceptance rate: ", acceptance(chain2, diagnostics=false))

theta = hcat(chain1.value, chain2.value);
for i in sample(1:length(theta), 200)
  @test isapprox(h0(theta[i])/M0, h1(theta[i])/M1)
end

# plot(x=chain1.value, Geom.histogram(bincount=20))

# plot(x=chain2.value, Geom.histogram(bincount=20))
