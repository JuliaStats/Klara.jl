using Distributions, MCMC

mcmodel = model(Normal(), init=1.)

mcchain = run(mcmodel, HMC(1.25), SerialMC(steps=10000, burnin=1000))

acceptance(mcchain)

describe(mcchain)
