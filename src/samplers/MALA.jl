### HMCBaseStash type holds the internal state ("local variables") of the HMC sampler

type MALAStash <: MCStash{LMCBaseSampler, MCGradSample}
  instate::MCState{MCGradSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCGradSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  driftstep::Float64 # Drift stepsize for the current Monte Carlo iteration (possibly tuned)
end

MALAStash() = MALAStash(MCGradSample(), MCGradSample(), VanillaMCTune(), 0, NaN)

MALAStash(l::Int) =
  MALAStash(MCState(MCGradSample(l), MCGradSample(l)), MCState(MCGradSample(l), MCGradSample(l)), VanillaMCTune(), 0,
  NaN)
