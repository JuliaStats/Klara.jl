using Distributions, MCMC, Optim
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
res0 = optimize(h0, ones(L))
M0 = -res0.f_minimum
log_M0 = log(M0)

# Candidate (starting) Distribution, use Normal(0, 2)
g1(x) = pdf(Normal(0, 2), x)
log_g1(x) = sum(logpdf(Normal(0, 2), x))

# Compute M1 such that candidate dominates target
h1(x) = -g(x)/sum(g1(x))
res1 = optimize(h1, ones(L))
M1 = -res1.f_minimum
log_M1 = log(M1)

mcmodel = model(log_g, init=ones(L))

mcchain = run(mcmodel, [ARS(log_g0, log_M0), ARS(log_g1, log_M1)], SerialMC(1000:1:10000))

println("Acceptance rate: ", acceptance(mcchain[1]))
describe(mcchain[1])

println("Acceptance rate: ", acceptance(mcchain[2]))
describe(mcchain[2])

w = hcat(mcchain[1].diagnostics["weight"]', mcchain[2].diagnostics["weight"]');
theta = hcat(mcchain[1].samples, mcchain[2].samples);
for i in sample(1:length(theta), 200)
  #println([theta[i] g(theta[i]) g0(theta[i]) g1(theta[i]) -h0(theta[i])/M0 -h1(theta[i])/M1 w[i]])
  @test_approx_eq h0(theta[i])/M0 h1(theta[i])/M1
end
