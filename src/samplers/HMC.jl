### The HMCBaseSampler holds the fields that fully define a HMC sampler
### These fields represent the initial user-defined state of the sampler
### These sampler fields are copied to the corresponding stash type fields, where the latter can be tuned

immutable HMCBaseSampler <: HMCSampler
  nleaps::Int # Number of leapfrog steps
  leapstep::Float64 # Leapfrog stepsize

  function HMCBaseSampler(nl::Int, l::Float64)
    @assert nl > 0 "Number of leapfrog steps is not positive."
    @assert l > 0 "Leapfrog stepsize is not positive."
    new(nl, l)
  end
end

HMCBaseSampler(nl::Int) = HMCBaseSampler(nl, 0.1)
HMCBaseSampler(l::Float64) = HMCBaseSampler(10, l)

HMCBaseSampler(; nleaps::Int=10, leapstep::Float64=0.1) = HMCBaseSampler(nleaps, leapstep)

typealias HMC HMCBaseSampler

### HMCBaseStash type holds the internal state ("local variables") of the HMC sampler

type HMCBaseStash <: MCStash
  instate::MCState # Monte Carlo state used internally by the sampler
  outstate::MCState # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  nleaps::Int
  leapstep::Float64
end

# HMCBaseStash() = HMCBaseStash(MCState(), MCState(), VanillaMCTune(), 0, 0, NaN)
