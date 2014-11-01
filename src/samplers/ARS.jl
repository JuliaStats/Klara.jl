### ARS holds the fields of the acceptance-rejection sampler

immutable ARS <: MCSampler
  logproposal::Function # Possibly unnormalized log-proposal
  proposalscale::Float64 # Scale factor to ensure the scaled-up logproposal covers target
  jumpscale::Float64 # Scale factor for adapting the jump size

  function ARS(l::Function, ps::Float64, js::Float64)
    @assert typeof(l) == Function "logproposal should be a function."
    @assert js > 0 "Scale factor for adapting the jump size is not positive."
    new(l, ps, js)
  end
end

ARS(l::Function) = ARS(l, 1., 1.)
ARS(l::Function, ps::Float64) = ARS(l, ps, 1.)

ARS(l::Function; proposalscale::Float64 = 1.0, jumpscale::Float64=1.0) = ARS(l, proposalscale, jumpscale)

### ARSStash type holds the internal state ("local variables") of the ARS sampler

type ARSStash <: MCStash{MCBaseSample}
  instate::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCBaseSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  jumpscale::Vector{Float64}
  logproposal::Float64
  weight::Float64
end

ARSStash() =
  ARSStash(MCState(MCBaseSample(), MCBaseSample()), MCState(MCBaseSample(), MCBaseSample()), VanillaMCTune(), 0,
  Float64[], NaN, NaN)

ARSStash(l::Int) =
  ARSStash(MCState(MCBaseSample(l), MCBaseSample(l)), MCState(MCBaseSample(l), MCBaseSample(l)), VanillaMCTune(), 0,
  Float64[], NaN, NaN)

### Initialize Metropolis-Hastings sampler

function initialize_stash(m::MCModel, s::ARS, r::MCRunner, t::MCTuner)
  stash::ARSStash = ARSStash(m.size)

  stash.instate.current = MCBaseSample(copy(m.init))
  logtarget!(stash.instate.current, m.eval)
  @assert isfinite(stash.instate.current.logtarget) "Initial values out of model support."

  stash.jumpscale = m.scale.*s.jumpscale  # rescale model scale by sampler scale

  stash.tune = VanillaMCTune()

  stash.count = 1

  stash
end

function reset!(stash::ARSStash, x::Vector{Float64})
  stash.instate.current = MCBaseSample(copy(x))
  logtarget!(stash.instate.current, m.eval)
end

function initialize_task!(stash::ARSStash, m::MCModel, s::ARS, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(stash, x))

  while true
    iterate!(stash, m, s, r, t, produce)
  end
end

### Perform iteration for ARS sampler

function iterate!(stash::ARSStash, m::MCModel, s::ARS, r::MCRunner, t::MCTuner, send::Function)
  if t.verbose
    stash.tune.proposed += 1
  end

  stash.instate.successive = MCBaseSample(stash.instate.current.sample+randn(m.size).*stash.jumpscale)
  logtarget!(stash.instate.successive, m.eval)
  stash.logproposal = s.logproposal(stash.instate.successive.sample)

  stash.weight = stash.instate.successive.logtarget.-s.proposalscale.-stash.logproposal
  if stash.weight > log(rand())
    stash.outstate = MCState(stash.instate.successive, stash.instate.current, {"accept" => true})
    stash.instate.current = deepcopy(stash.instate.successive)

    if t.verbose
      stash.tune.accepted += 1
    end
  else
    stash.outstate = MCState(stash.instate.current, stash.instate.current, {"accept" => false})
  end

  if t.verbose && stash.count <= r.burnin && mod(stash.count, t.period) == 0
    rate!(stash.tune)
    println("Burnin iteration $(stash.count) of $(r.burnin): ", round(100*stash.tune.rate, 2), " % acceptance rate")
  end

  stash.count += 1

  send(stash.outstate)
end
