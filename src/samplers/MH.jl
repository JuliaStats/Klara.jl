### MH holds the fields of the Metropolis-Hastings sampler
### In its most general case it accommodates an asymmetric proposal density
### For symetric proposals, the proposal correction factor equals 1, so the logproposal field is set to nothing

immutable MH <: MHSampler
  symmetric::Bool # If the proposal density is symmetric, then symmetric=true, otherwise symmetric=false
  logproposal::Union(Function, Nothing) # logpdf of asymmetric proposal density
                                        # For symmetric proposals, logproposal is set to nothing
  randproposal::Function # random sampling from proposal density

  function MH(s::Bool, l::Union(Function, Nothing), r::Function)
    if s && !isa(l, Nothing)
      error("If the symmetric field is true, then logproposal is not used in the calculations.")
    end
    new(s, l, r)
  end
end

MH(l::Function, r::Function) = MH(false, l, r) # Metropolis-Hastings sampler (asymmetric proposal)
MH(r::Function) = MH(true, nothing, r) # Metropolis sampler (symmetric proposal)

# Random-walk Metropolis, i.e. Metropolis with a normal proposal density
MH(σ::Vector{Float64}) = MH(x::Vector{Float64} -> rand(DiagNormal(x, σ)))
MH(σ::Float64) = MH(x::Vector{Float64} -> rand(IsoNormal(x, σ)))
MH() = MH(x::Vector{Float64} -> rand(IsoNormal(x, 1.)))

### MHStash type holds the internal state ("local variables") of the MALA sampler

type MHStash <: MCStash{MCBaseSample}
  instate::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCBaseSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  ratio::Float64
end

MHStash() =
  MHStash(MCState(MCBaseSample(), MCBaseSample()), MCState(MCBaseSample(), MCBaseSample()), VanillaMCTune(), 0, NaN)

MHStash(l::Int, t::MCTune=VanillaMCTune()) =
  MHStash(MCState(MCBaseSample(l), MCBaseSample(l)), MCState(MCBaseSample(l), MCBaseSample(l)), t, 0, NaN)

### Initialize Metropolis-Hastings sampler

function initialize(m::MCModel, s::MH, r::MCRunner, t::MCTuner)
  stash::MHStash = MHStash(m.size)

  stash.instate.current = MCBaseSample(copy(m.init))
  logtarget!(stash.instate.current, m.eval)
  @assert isfinite(stash.instate.current.logtarget) "Initial values out of model support."

  stash.tune = VanillaMCTune()

  stash.count = 1

  stash
end

function initialize_task(m::MCModel, s::MH, r::MCRunner, t::MCTuner)
  stash::MHStash = initialize(m, s, r, t)

  # Hook inside Task to allow remote resetting
  task_local_storage(:reset,
    (x::Vector{Float64}) ->
    (stash.instate.current = MCBaseSample(copy(x)); logtarget!(stash.instate.current, m.eval))) 

  while true
    iterate!(stash, m, s, r, t, produce)
  end
end

### Perform iteration for Metropolis-Hastings sampler

function iterate!(stash::MHStash, m::MCModel, s::MH, r::MCRunner, t::MCTuner, send::Function)
  if t.verbose
    stash.tune.proposed += 1
  end

  stash.instate.successive = MCBaseSample(s.randproposal(stash.instate.current.sample))
  logtarget!(stash.instate.successive, m.eval)

  if s.symmetric
    stash.ratio = stash.instate.successive.logtarget-stash.instate.current.logtarget
  else
    stash.ratio = stash.instate.successive.logtarget
      +s.logproposal(stash.instate.successive.sample, stash.instate.current.sample)
      -stash.instate.current.logtarget
      -s.logproposal(stash.instate.current.sample, stash.instate.successive.sample)
  end
  if stash.ratio > 0 || (stash.ratio > log(rand()))
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
