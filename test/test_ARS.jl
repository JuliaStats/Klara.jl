using Distributions, MCMC, Optim

# Target distribution shape, unscaled
g(x) = exp(-dot(x, x)/2)
log_g(x) = -dot(x, x)/2

# Candidate (starting) density, use Normal(0, 2)
g0(x) = 1/(sqrt(2*pi)*2)*exp(-dot(x, x)/8)
log_g0(x) = -dot(x, x)/8-log(2 * sqrt(2*pi))

# Compute M such that candidate dominates target
h(x) = -g(x)/g0(x)
res = optimize(h,ones(2))
M = - res.f_minimum
log_M = log(M)

mcmodel = model(log_g, init=ones(2))

mcchain = run(mcmodel, ARS(log_g0, log_M), SerialMC(1000:10:10000))

println("Acceptance rate: ", acceptance(mcchain))
describe(mcchain)

mcchain01 = deepcopy(mcchain)
indx = find(mcchain.diagnostics["accept"])
mcchain01.samples = mcchain.samples[indx,:]
for (k, v) in mcchain.diagnostics
  mcchain01.diagnostics[k] = mcchain.diagnostics[k][indx]
end

l = filter((el)->el==true, mcchain.diagnostics["accept"])
describe(mcchain01)
