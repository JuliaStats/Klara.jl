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

### HMCBaseStash type holds the internal state ("local variables") of the HMC sampler

type HMCStash <: MCStash{HMCSample}
  instate::MCState{HMCSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{HMCSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  nleaps::Int # Number of leapfrog steps for the current Monte Carlo iteration (possibly tuned)
  leapstep::Float64 # Leapfrog stepsize for the current Monte Carlo iteration (possibly tuned)
end

HMCStash() = HMCStash(HMCSample(), HMCSample(), VanillaMCTune(), 0, 0, NaN)

HMCStash(l::Int) =
  HMCStash(MCState(HMCSample(l), HMCSample(l)), MCState(HMCSample(l), HMCSample(l)), VanillaMCTune(), 0, 0, NaN)

### Initialize HMC sampler

function initialize(model::MCModel, sampler::HMC, runner::MCRunner, tuner::MCTuner)
  @assert hasgradient(model) "HMC sampler requires model with gradient function."
  stash::HMCStash = HMCStash(model.size)

  stash.instate.current = HMCSample(copy(model.init))
  gradlogtargetall!(stash.instate.current, model.evalallg)
  @assert isfinite(stash.instate.current.logtarget) "Initial values out of model support."

  if isa(tuner, VanillaMCTuner)
    stash.tune = VanillaMCTune()
  elseif isa(tuner, EmpiricalMCTuner)
    stash.tune = EmpiricalHMCTune(sampler.leapstep, sampler.nleaps)
  end

  stash.count = 1

  stash
end

function initialize_task(model::MCModel, sampler::HMC, runner::MCRunner, tuner::MCTuner)
  stash::HMCStash = initialize(model, sampler, runner, tuner)

  # Hook inside Task to allow remote resetting
  task_local_storage(:reset,
    (x::Vector{Float64}) ->
    (stash.instate.current = HMCSample(copy(x)); gradlogtargetall!(stash.instate.current, model.evalallg))) 

  while true
    iterate!(stash, model, sampler, runner, tuner, produce)
  end
end

### Perform iteration for HMC sampler

function iterate!(stash::HMCStash, model::MCModel, sampler::HMC, runner::MCRunner, tuner::MCTuner, send::Function)
  if isa(tuner, VanillaMCTuner)
    stash.nleaps, stash.leapstep = sampler.nleaps, sampler.leapstep
    if tuner.verbose
      stash.tune.proposed += 1
    end
  elseif isa(tuner, EmpiricalMCTuner)
    stash.nleaps, stash.leapstep = stash.tune.nsteps, stash.tune.step
    stash.tune.proposed += 1
  end

  stash.instate.current.momentum = randn(model.size)
  hamiltonian!(stash.instate.current)
  stash.instate.successive = deepcopy(stash.instate.current)

  for j = 1:stash.nleaps
    stash.instate.successive = leapfrog(stash.instate.successive, model.evalallg, stash.leapstep)
  end

  if rand() < exp(stash.instate.current.hamiltonian-stash.instate.successive.hamiltonian)
    stash.outstate = MCState(stash.instate.successive, stash.instate.current, {"accept" => true})
    stash.instate.current = deepcopy(stash.instate.successive)

    if isa(tuner, VanillaMCTuner) && tuner.verbose
      stash.tune.accepted += 1 
    elseif isa(tuner, EmpiricalMCTuner)
      stash.tune.accepted += 1     
    end
  else
    stash.outstate = MCState(stash.instate.current, stash.instate.current, {"accept" => false})
  end

  if isa(tuner, VanillaMCTuner) && tuner.verbose && stash.count <= runner.burnin && mod(stash.count, tuner.period) == 0
    rate!(stash.tune)
    println("Burnin iteration $(stash.count) of $(runner.burnin): ", round(100*stash.tune.rate, 2),
      " % acceptance rate")
  elseif isa(tuner, EmpiricalMCTuner) && stash.count <= runner.burnin && mod(stash.count, tuner.period) == 0
    adapt!(stash.tune, tuner)
    reset!(stash.tune)
    if tuner.verbose
      println("Burnin iteration $(stash.count) of $(runner.burnin): ", round(100*stash.tune.rate, 2),
        " % acceptance rate")
    end
  end

  stash.count += 1

  send(stash.outstate)
end
