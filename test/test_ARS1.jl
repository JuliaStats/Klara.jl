using Distributions, MCMC

# Using Optim, M0 and M1 would be ~ 5.01, acceptance rate ~ 60%
#using Optim

using Base.Test
srand(1)

L = 1

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

mcmodel = model(log_g, init=ones(L))

mcchain = run(MCSystem(fill(mcmodel, 2), [ARS(log_g0, log_M0), ARS(log_g1, log_M1)], fill(SerialMC(1000:1:10000), 2)))

println()
println("M: ", [M0 M1])
println()

println("Acceptance rate: ", acceptance(mcchain[1]))
describe(mcchain[1])

println("Acceptance rate: ", acceptance(mcchain[2]))
describe(mcchain[2])

theta = hcat(mcchain[1].samples, mcchain[2].samples);
for i in sample(1:length(theta), 200)
  @test_approx_eq h0(theta[i])/M0 h1(theta[i])/M1
end
