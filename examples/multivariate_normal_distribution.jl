using Distributions, MCMC

mcmodel = model(MvNormal([1. 0.1; 0.1 1]), init=[1., 2.])

mcchain = run(mcmodel, HMC(0.75), SerialMC(steps=10000, burnin=1000))

acceptance(mcchain)

describe(mcchain)
