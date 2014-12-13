using Distributions, PDMats, Lora

# Using Optim, M would be ~ 106, acceptance rate ~ 12%
#using Optim

using Base.Test
#srand(1)

m = [0.0, 2.0]
sig = 0.4
mu = [1.0, 1.0]
sigma = 4

# Target distribution shape, unscaled
d = DiagNormal(m, PDiagMat(sig^2 * ones(2)))
g(x) = exp(sum(logpdf(d, x)))
log_g(x) = sum(logpdf(d, x))

# Candidate (starting) Distribution, use MvNormal([1, 1], [1 0; 0 1])
d0 = DiagNormal(mu, PDiagMat(sigma^2 * ones(2)))
g0(x) = exp(sum(logpdf(d0, x)))
log_g0(x) = sum(logpdf(d0, x))

# Compute M such that candidate dominates target
h(x) = -g(x)/g0(x)
#res = optimize(h, ones(2))
#M = -res.f_minimum
M = 120.0
log_M = log(M)

mcmodel = model(log_g, init=ones(2))

mcchain = run(mcmodel, ARS(log_g0, log_M), SerialMC(1000:1:10000))

println()
println("M: ", M)
println()

println("Acceptance rate g0: ", acceptance(mcchain))
println()
describe(mcchain)
