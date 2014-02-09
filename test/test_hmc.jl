using MCMC

println("    Testing basic HMC constructors...")

HMC()
HMC(20)
HMC(0.75)
HMC(20, 0.75)
HMC(init=20)
HMC(scale=0.75)
HMC(init=20, scale=0.75)
mctuner = EmpMCTuner(0.85)
HMC(mctuner)
HMC(20, mctuner)
HMC(0.75, mctuner)
HMC(20, 0.75, mctuner)
HMC(init=20, tuner=mctuner)
HMC(scale=0.75, tuner=mctuner)
HMC(init=20, scale=0.75, tuner=mctuner)

println("    Testing that HMC sampler can run...")

mcmodel = model(v->-dot(v,v), grad=v->-2v, init=ones(3)) 
mcchain = run(mcmodel, HMC(0.75), SerialMC(steps=5000, burnin=1000))
# mcchain = run(mcmodel, HMC(0.75, mctuner), SerialMC(steps=5000, burnin=1000))
