### SMMALA holds the fields of the SMMALA sampler
### These fields represent the initial user-defined state of the sampler
### These sampler fields are copied to the corresponding stash type fields, where the latter can be tuned

immutable SMMALA <: LMCSampler
  driftstep::Float64

  function SMMALA(ds::Float64)
    @assert ds > 0 "Drift step is not positive."
    new(ds)
  end
end

SMMALA(; driftstep::Float64=1.) = SMMALA(driftstep)

### SMMALAStash type holds the internal state ("local variables") of the SMMALA sampler

type SMMALAStash <: MCStash{MCTensorSample}
  instate::MCState{MCTensorSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCTensorSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  driftstep::Float64 # Drift stepsize for the current Monte Carlo iteration (possibly tuned)
  smean::Vector{Float64}
  ratio::Float64
  pnewgivenold::Float64
  poldgivennew::Float64
  current_invtensor::Matrix{Float64}
  successive_invtensor::Matrix{Float64}
  current_termone::Vector{Float64}
  successive_termone::Vector{Float64}
  cholinvtensor
end

SMMALAStash() =
  SMMALAStash(MCState(MCTensorSample(), MCTensorSample()), MCState(MCTensorSample(), MCTensorSample()), VanillaMCTune(),
  0, NaN, Float64[], NaN, NaN, NaN, Array(Float64, 0, 0), Array(Float64, 0, 0), Float64[], Float64[],
  Triangular(Array(Float64, 0, 0), :U))

SMMALAStash(l::Int, t::MCTune=VanillaMCTune()) =
  SMMALAStash(MCState(MCTensorSample(l), MCTensorSample(l)), MCState(MCTensorSample(l), MCTensorSample(l)), t, 0, NaN,
    fill(NaN, l), NaN, NaN, NaN, fill(NaN, l, l), fill(NaN, l, l), fill(NaN, l), fill(NaN, l),
    Triangular(fill(NaN, l, l), :U))

### Initialize SMMALA sampler

function initialize(m::MCModel, s::SMMALA, r::MCRunner, t::MCTuner)
  @assert hasgradient(m) "SMMALA sampler requires model with gradient function."
  @assert hastensor(m) "SMMALA sampler requires model with tensor function."
  stash::SMMALAStash = SMMALAStash(m.size)

  stash.instate.current = MCTensorSample(copy(m.init))
  tensorlogtargetall!(stash.instate.current, m.evalallt)
  @assert isfinite(stash.instate.current.logtarget) "Initial values out of model support."

  stash.current_invtensor = inv(stash.instate.current.tensorlogtarget)
  stash.current_termone = stash.current_invtensor*stash.instate.current.gradlogtarget

  if isa(t, VanillaMCTuner)
    stash.tune = VanillaMCTune()
  elseif isa(t, EmpiricalMCTuner)
    stash.tune = EmpiricalMCTune(s.driftstep)
  end

  stash.count = 1

  stash
end

function reset!(stash::SMMALAStash, x::Vector{Float64})
  stash.instate.current = MCTensorSample(copy(x))
  tensorlogtargetall!(stash.instate.current, m.evalallt)
end

function initialize_task(m::MCModel, s::SMMALA, r::MCRunner, t::MCTuner)
  stash::SMMALAStash = initialize(m, s, r, t)

  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(stash, x))

  while true
    iterate!(stash, m, s, r, t, produce)
  end
end

### Perform iteration for SMMALA sampler

function iterate!(stash::SMMALAStash, m::MCModel, s::SMMALA, r::MCRunner, t::MCTuner, send::Function)
  if isa(t, VanillaMCTuner)
    stash.driftstep = s.driftstep
    if t.verbose
      stash.tune.proposed += 1
    end
  elseif isa(t, EmpiricalMCTuner)
    stash.driftstep = stash.tune.step
    stash.tune.proposed += 1
  end

  # Calculate the drift term
  stash.smean = stash.instate.current.sample+(stash.driftstep/2)*stash.current_termone

  # Calculate proposed parameters
  stash.cholinvtensor = chol(stash.driftstep*stash.current_invtensor)
  stash.instate.successive = MCTensorSample(stash.smean+stash.cholinvtensor'*randn(m.size))

  # Update model based on the proposed parameters
  tensorlogtargetall!(stash.instate.successive, m.evalallt)

  stash.pnewgivenold = (-sum(log(diag(stash.cholinvtensor)))
    -(0.5*(stash.smean-stash.instate.successive.sample)'*(stash.instate.current.tensorlogtarget/stash.driftstep)
    *(stash.smean-stash.instate.successive.sample))[1])

  stash.successive_invtensor = inv(stash.instate.successive.tensorlogtarget)
  stash.successive_termone = stash.successive_invtensor*stash.instate.successive.gradlogtarget

  # Calculate the drift term
  stash.smean = stash.instate.successive.sample+(stash.driftstep/2)*stash.successive_termone

  stash.poldgivennew = (-sum(log(diag(chol(stash.driftstep*eye(m.size)*stash.successive_invtensor))))
    -(0.5*(stash.smean-stash.instate.current.sample)'*(stash.instate.successive.tensorlogtarget/stash.driftstep)
    *(stash.smean-stash.instate.current.sample))[1])
 
  stash.ratio = stash.instate.successive.logtarget+stash.poldgivennew-stash.instate.current.logtarget-stash.pnewgivenold

  if stash.ratio > 0 || (stash.ratio > log(rand()))
    stash.outstate = MCState(stash.instate.successive, stash.instate.current, Dict{Any, Any}("accept" => true))
    stash.instate.current = deepcopy(stash.instate.successive)
    stash.current_invtensor, stash.current_termone = copy(stash.successive_invtensor), copy(stash.successive_termone)
      
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
