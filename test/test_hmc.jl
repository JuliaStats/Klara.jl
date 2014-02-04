using MCMC

mcmodel = model(v-> -dot(v,v), grad=v->-2v, init=ones(3)) 

mcchain = run(mcmodel, HMC(0.75), SerialMC(steps=10000, burnin=1000))
mcchain = run(mcmodel * HMC(20, 0.75) * SerialMC(1000:10000))
