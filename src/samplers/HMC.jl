### HMC holds the fields that fully define a HMC sampler
### These fields represent the initial user-defined state of the sampler
### These sampler fields are copied to the corresponding stash type fields, where the latter can be tuned

immutable HMC <: HMCSampler
  nleaps::Int # Number of leapfrog steps
  leapstep::Float64 # Leapfrog stepsize

  function HMC(nl::Int, l::Float64)
    @assert nl > 0 "Number of leapfrog steps is not positive."
    @assert l > 0 "Leapfrog stepsize is not positive."
    new(nl, l)
  end
end

HMC(nl::Int) = HMC(nl, 0.1)
HMC(l::Float64) = HMC(10, l)

HMC(; nleaps::Int=10, leapstep::Float64=0.1) = HMC(nleaps, leapstep)

### Sample type for HMC

type HMCSample <: MCSample{FirstOrder}
  sample::Vector{Float64} # sample position, i.e. model parameters
  logtarget::Float64
  gradlogtarget::Vector{Float64}
  momentum::Vector{Float64}
  hamiltonian::Float64
end

HMCSample(s::Vector{Float64}) = HMCSample(s, NaN, Float64[], Float64[], NaN)
HMCSample() = HMCSample(Float64[], NaN, Float64[], Float64[], NaN)

HMCSample(l::Int) = HMCSample(fill(NaN, l), NaN, fill(NaN, l), fill(NaN, l), NaN)

hamiltonian!(s::HMCSample) = (s.hamiltonian = -s.logtarget+0.5*dot(s.momentum, s.momentum))

function leapfrog(s::HMCSample, f::Function, leapstep::Float64)
  lsample = deepcopy(s)
  lsample.momentum += 0.5*lsample.gradlogtarget*leapstep
  lsample.sample += leapstep*lsample.momentum
  gradlogtargetall!(lsample, f)
  lsample.momentum += 0.5*lsample.gradlogtarget*leapstep
  hamiltonian!(lsample)
  lsample
end

### HMCStash type holds the internal state ("local variables") of the HMC sampler

type HMCStash <: MCStash{HMCSample}
  instate::MCState{HMCSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{HMCSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  nleaps::Int # Number of leapfrog steps for the current Monte Carlo iteration (possibly tuned)
  leapstep::Float64 # Leapfrog stepsize for the current Monte Carlo iteration (possibly tuned)
end

HMCStash() = HMCStash(MCState(HMCSample(), HMCSample()), MCState(HMCSample(), HMCSample()), VanillaMCTune(), 0, 0, NaN)

HMCStash(l::Int, t::MCTune=VanillaMCTune()) =
  HMCStash(MCState(HMCSample(l), HMCSample(l)), MCState(HMCSample(l), HMCSample(l)), t, 0, 0, NaN)

### Initialize HMC sampler

function initialize(m::MCModel, s::HMC, r::MCRunner, t::MCTuner)
  @assert hasgradient(m) "HMC sampler requires model with gradient function."
  stash::HMCStash = HMCStash(m.size)

  stash.instate.current = HMCSample(copy(m.init))
  gradlogtargetall!(stash.instate.current, m.evalallg)
  @assert isfinite(stash.instate.current.logtarget) "Initial values out of model support."

  if isa(t, VanillaMCTuner)
    stash.tune = VanillaMCTune()
  elseif isa(t, EmpiricalMCTuner)
    stash.tune = EmpiricalHMCTune(s.leapstep, s.nleaps)
  end

  stash.count = 1

  stash
end

function initialize_task(m::MCModel, s::HMC, r::MCRunner, t::MCTuner)
  stash::HMCStash = initialize(m, s, r, t)

  # Hook inside Task to allow remote resetting
  task_local_storage(:reset,
    (x::Vector{Float64}) ->
    (stash.instate.current = HMCSample(copy(x)); gradlogtargetall!(stash.instate.current, m.evalallg))) 

  while true
    iterate!(stash, m, s, r, t, produce)
  end
end

### Perform iteration for HMC sampler

function iterate!(stash::HMCStash, m::MCModel, s::HMC, r::MCRunner, t::MCTuner, send::Function)
  if isa(t, VanillaMCTuner)
    stash.nleaps, stash.leapstep = s.nleaps, s.leapstep
    if t.verbose
      stash.tune.proposed += 1
    end
  elseif isa(t, EmpiricalMCTuner)
    stash.nleaps, stash.leapstep = stash.tune.nsteps, stash.tune.step
    stash.tune.proposed += 1
  end

  stash.instate.current.momentum = randn(m.size)
  hamiltonian!(stash.instate.current)
  stash.instate.successive = deepcopy(stash.instate.current)

  for j = 1:stash.nleaps
    stash.instate.successive = leapfrog(stash.instate.successive, m.evalallg, stash.leapstep)
  end

  if rand() < exp(stash.instate.current.hamiltonian-stash.instate.successive.hamiltonian)
    stash.outstate = MCState(stash.instate.successive, stash.instate.current, {"accept" => true})
    stash.instate.current = deepcopy(stash.instate.successive)

    if isa(t, VanillaMCTuner) && t.verbose
      stash.tune.accepted += 1 
    elseif isa(t, EmpiricalMCTuner)
      stash.tune.accepted += 1     
    end
  else
    stash.outstate = MCState(stash.instate.current, stash.instate.current, {"accept" => false})
  end

  if isa(t, VanillaMCTuner) && t.verbose && stash.count <= r.burnin && mod(stash.count, t.period) == 0
    rate!(stash.tune)
    println("Burnin iteration $(stash.count) of $(r.burnin): ", round(100*stash.tune.rate, 2), " % acceptance rate")
  elseif isa(t, EmpiricalMCTuner) && stash.count <= r.burnin && mod(stash.count, t.period) == 0
    adapt!(stash.tune, t)
    reset!(stash.tune)
    if t.verbose
      println("Burnin iteration $(stash.count) of $(r.burnin): ", round(100*stash.tune.rate, 2), " % acceptance rate")
    end
  end

  stash.count += 1

  send(stash.outstate)
end
