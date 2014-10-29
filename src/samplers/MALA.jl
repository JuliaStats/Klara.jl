### MALA holds the fields of the MALA sampler
### These fields represent the initial user-defined state of the sampler
### These sampler fields are copied to the corresponding stash type fields, where the latter can be tuned

immutable MALA <: LMCSampler
  driftstep::Float64

  function MALA(ds::Float64)
    @assert ds > 0 "Drift step is not positive."
    new(ds)
  end
end

MALA(; driftstep::Float64=1.) = MALA(driftstep)

### MALAStash type holds the internal state ("local variables") of the MALA sampler

type MALAStash <: MCStash{MCGradSample}
  instate::MCState{MCGradSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCGradSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  driftstep::Float64 # Drift stepsize for the current Monte Carlo iteration (possibly tuned)
  smean::Vector{Float64}
  ratio::Float64
  pnewgivenold::Float64
  poldgivennew::Float64
end

MALAStash() =
  MALAStash(MCState(MCGradSample(), MCGradSample()), MCState(MCGradSample(), MCGradSample()), VanillaMCTune(), 0, NaN,
  Float64[], NaN, NaN, NaN)

MALAStash(l::Int, t::MCTune=VanillaMCTune()) =
  MALAStash(MCState(MCGradSample(l), MCGradSample(l)), MCState(MCGradSample(l), MCGradSample(l)), t, 0, NaN,
    fill(NaN, l), NaN, NaN, NaN)

### Initialize MALA sampler

function initialize_stash(m::MCModel, s::MALA, r::MCRunner, t::MCTuner)
  @assert hasgradient(m) "MALA sampler requires model with gradient function."
  stash::MALAStash = MALAStash(m.size)

  stash.instate.current = MCGradSample(copy(m.init))
  gradlogtargetall!(stash.instate.current, m.evalallg)
  @assert isfinite(stash.instate.current.logtarget) "Initial values out of model support."

  if isa(t, VanillaMCTuner)
    stash.tune = VanillaMCTune()
  elseif isa(t, EmpiricalMCTuner)
    stash.tune = EmpiricalMCTune(s.driftstep)
  end

  stash.count = 1

  stash
end

function reset!(stash::MALAStash, x::Vector{Float64})
  stash.instate.current = MCGradSample(copy(x))
  gradlogtargetall!(stash.instate.current, m.evalallg)
end

function initialize_task!(stash::MALAStash, m::MCModel, s::MALA, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(stash, x)) 

  while true
    iterate!(stash, m, s, r, t, produce)
  end
end

### Perform iteration for MALA sampler

function iterate!(stash::MALAStash, m::MCModel, s::MALA, r::MCRunner, t::MCTuner, send::Function)
  if isa(t, VanillaMCTuner)
    stash.driftstep = s.driftstep
    if t.verbose
      stash.tune.proposed += 1
    end
  elseif isa(t, EmpiricalMCTuner)
    stash.driftstep = stash.tune.step
    stash.tune.proposed += 1
  end

  stash.smean = stash.instate.current.sample+(stash.driftstep/2.)*stash.instate.current.gradlogtarget
  stash.instate.successive = MCGradSample(stash.smean+sqrt(stash.driftstep)*randn(m.size))
  gradlogtargetall!(stash.instate.successive, m.evalallg)
  stash.pnewgivenold =
    sum(-(stash.smean-stash.instate.successive.sample).^2/(2*stash.driftstep).-log(2*pi*stash.driftstep)/2)

  stash.smean = stash.instate.successive.sample+(stash.driftstep/2)*stash.instate.successive.gradlogtarget
  stash.poldgivennew =
    sum(-(stash.smean-stash.instate.current.sample).^2/(2*stash.driftstep).-log(2*pi*stash.driftstep)/2)
    
  stash.ratio = stash.instate.successive.logtarget+stash.poldgivennew-stash.instate.current.logtarget-stash.pnewgivenold
  if stash.ratio > 0 || (stash.ratio > log(rand()))
    stash.outstate = MCState(stash.instate.successive, stash.instate.current, Dict{Any, Any}("accept" => true))
    stash.instate.current = deepcopy(stash.instate.successive)

    if isa(t, VanillaMCTuner) && t.verbose
      stash.tune.accepted += 1 
    elseif isa(t, EmpiricalMCTuner)
      stash.tune.accepted += 1     
    end
  else
    stash.outstate = MCState(stash.instate.current, stash.instate.current, Dict{Any, Any}("accept" => false))
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
