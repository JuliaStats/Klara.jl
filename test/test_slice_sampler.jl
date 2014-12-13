# TODO: this file will be used for setting up similar tests for the rest of the samplers.
#
# This samples the funnel distribution, then tests that inference is correct.
#

using Lora

#log  ~ N(0, 3), x_i ~ N(0, ), i=1,2,...,model.size

function funnel_loglik(x::Vector{Float64})
    # First element is log(sigma^2). Rest of the elements are x_d
    s2 = exp(x[1])
    nx = length(x) - 1
    -0.5*((x[1]-0.0)^2/9.0 + dot(x[2:end], x[2:end])/s2 + nx*log(s2))
end

mcmodel = model(funnel_loglik, init=[10, zeros(4)])
mcsampler = SliceSampler()
mcrunner = SerialMC(nsteps=100000, burnin=10000)
mcchain = run(mcmodel, mcsampler, mcrunner)

samples01 = mcchain.samples[:, 1]

ess01 = ess(samples01, maxlag=1000)

@printf("Monte Carlo samples of first parameter: ess=%.2f, out of raw %d\n", ess01, length(samples01))

mean01 = mean(samples01)
std01 = std(samples01)
meanse01 = std01/sqrt(ess01)
zscore01 = mean01/meanse01

@printf("Monte Carlo samples of first parameter: mean=%f, std=%f, meanse=%f, zscore=%f\n",
  mean01, std01, meanse01, zscore01)
if abs(zscore01) > 3
    warn("Log-posterior mean estimate for the first parameter differs from truth by more than 3 MC-stderr. Chance of this should be <0.3%.")
end

var01 = var(samples01)
varse01 = var01*sqrt(2/(ess01-1))
zscore01 = (var01-9)/varse01

@printf("Monte Carlo samples of first parameter: var=%f, varse=%f, zscore=%f\n", var01, varse01, zscore01)
if abs(zscore01) > 3
    warn("Log-posterior mean estimate for the first parameter differs from truth by more than 3 MC-stderr. Chance of this should be <0.3%.")
end
