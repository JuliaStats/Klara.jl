using MCMC

println("    Testing basic EmpMCTuner constructors...")

mctuners = [EmpMCTuner(0.85),
  EmpMCTuner(0.85, adaptStep=50),
  EmpMCTuner(0.85, maxStep=100),
  EmpMCTuner(0.85, targetPath=0.75),
  EmpMCTuner(0.85, verbose=true)]

println("    Testing that EmpMCTuner tuners works with all samplers...")

npars = 3
mcmodel = model(v->-dot(v,v), grad=v->-2v, tensor=v->-2*ones(npars), dtensor=v->zeros(npars, npars, npars),
  init=ones(npars))

for mctuner in mctuners
  mcsamplers = [HMC(0.75, mctuner),
    RMHMC(0.75, mctuner),
    MALA(0.75, mctuner),
    SMMALA(0.75, mctuner),
    PMALA(0.75, mctuner)]

  for mcsampler in mcsamplers
    mcchain = run(mcmodel, HMC(0.75, mctuner), SerialMC(steps=5000, burnin=1000))
  end
end
