using MCMC

println("    Testing basic EmpiricalMCTuner constructors...")

mctuners = [EmpiricalMCTuner(0.85),
  EmpiricalMCTuner(0.85, period=50),
  EmpiricalMCTuner(0.85, maxnsteps=100),
  EmpiricalMCTuner(0.85, targetlen=1.25),
  EmpiricalMCTuner(0.85, verbose=true)]

println("    Testing that EmpiricalMCTuner tuners works with all samplers...")

npars = 3
mcmodel = model(v->-dot(v,v), grad=v->-2v, tensor=v->-2*ones(npars), dtensor=v->zeros(npars, npars, npars),
  init=ones(npars))

for mctuner in mctuners
  mcsamplers = [HMC(0.75), MALA(0.75)]#, SMMALA(0.75)]

  for mcsampler in mcsamplers
    mcchain = run(mcmodel, mcsampler, SerialMC(nsteps=5000, burnin=1000), mctuner)
  end
end
