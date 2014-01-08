# TODO this file should eventually go away and be folded into the
# infrastructure for all the other samplers.
#
# This samples the funnel distribution, then tests that the inferences are correct.
# Can trigger the warnings by modifying funnel_Lpstar; e.g. make the mean 0.8
# and it will detect the inferred mean is so far from 0 that there's a problem.
#
#Example output:
#julia> include("test_ss.jl")
#ls2 ess 2351.66, out of raw 99990 raw
#ls2 mean -0.011710, std 2.987608, meanse 0.061608, z -0.190079
#ls2 var 8.925801, varse 0.029169, z -0.000242


include("../src/samplers/slice_sample.jl")

#log σ² ~ N(0, 3²)
#x_d ~ N(0, σ²) d=1...D

function funnel_Lpstar(xx)
    # First element is log(sigma^2)
    # Rest of the elements are x_d
    s2 = exp(xx[1])
    nx = length(xx) - 1
    -0.5*((xx[1]-0.0)^2/9.0 + dot(xx[2:end], xx[2:end])/s2 + nx*log(s2))
end

# log(s2)=10 is a poor initializer; and 100 takes a long time to get out
samples = slice_sample(funnel_Lpstar, [10, zeros(4)], 100000)
ls2 = samples[11:end,1]
#using Winston; plot(samples[:,1])

##ess = var(ls2) / MCMC.mcvar_imse(ls2) # This is an attempt to do the same thing as ess.jl.  However, it somewhat disagrees with coda effectiveSize() but I don't know if it's supposed to or I'm doing it wrong

# This is a hack, but MCMC.mcvar_imse is slow for big N
writedlm("bla",ls2)
ess = readall(`Rscript -e 'library(coda); cat(effectiveSize(scan("bla")))'`)
ess = parsefloat(ess)

# This part should instead have an MCMCChain object so it
# can just use ESS, quantile, etc. other calculations as implemented in
# stats/*.jl

@printf("ls2 ess %.2f, out of raw %d raw\n", ess, length(ls2))

zscore = mean(ls2) / (std(ls2)/sqrt(ess))
@printf("ls2 mean %f, std %f, meanse %f, z %f\n", mean(ls2), std(ls2), std(ls2)/sqrt(ess), zscore)
if abs(zscore) > 3
    warn("log(s2) posterior mean estimate differs from truth by more than 3 MC-stderr. Chance of this should be <0.3%.")
end

# The z-scores seem too low here.  Is it wrong to substitute in ESS into the variance stderr equation?
varse = var(ls2) / sqrt(2/(ess-1))
zscore = (var(ls2)-9) / varse
@printf("ls2 var %f, varse %f, z %f\n", var(ls2), var(ls2)/varse, zscore)
if abs(zscore) > 3
    warn("log(s2) posterior variance estimate differs from truth by more than 3 MC-stderr. Chance of this should be <0.3%.")
end
