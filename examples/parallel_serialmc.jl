@everywhere using MCMC

@everywhere mymodel = model(v-> -dot(v,v), grad=v->-2v, init=ones(3))
@everywhere mytasks = mymodel * [HMC(0.75) for i in 1:10] * SerialMC(steps=50000, burnin=5000)

mychains = prun(mytasks)

[acceptance(chain) for chain in mychains]
