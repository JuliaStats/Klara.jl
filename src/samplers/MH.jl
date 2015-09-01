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
MH(σ::Vector{Float64}) = MH(x::Vector{Float64} -> rand(MvNormal(x, σ)))
MH(σ::Float64) = MH(x::Vector{Float64} -> rand(MvNormal(x, σ)))
MH() = MH(x::Vector{Float64} -> rand(MvNormal(x, 1.0)))

### MHHeap type holds the internal state ("local variables") of the MH sampler

type MHHeap <: MCHeap{MCBaseSample}
  instate::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCBaseSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  ratio::Float64
end

MHHeap() =
  MHHeap(MCState(MCBaseSample(), MCBaseSample()), MCState(MCBaseSample(), MCBaseSample()), VanillaMCTune(), 0, NaN)

MHHeap(l::Int, t::MCTune=VanillaMCTune()) =
  MHHeap(MCState(MCBaseSample(l), MCBaseSample(l)), MCState(MCBaseSample(l), MCBaseSample(l)), t, 0, NaN)

### Initialize Metropolis-Hastings sampler

function initialize_heap(m::MCModel, s::MH, r::MCRunner, t::MCTuner)
  heap::MHHeap = MHHeap(m.size)

  heap.instate.current = MCBaseSample(copy(m.init))
  logtarget!(heap.instate.current, m.eval)
  @assert isfinite(heap.instate.current.logtarget) "Initial values out of model support."

  heap.tune = VanillaMCTune()

  heap.count = 1

  heap
end

function reset!(heap::MHHeap, x::Vector{Float64}, m::MCModel)
  heap.instate.current = MCBaseSample(copy(x))
  logtarget!(heap.instate.current, m.eval)
end

function initialize_task!(heap::MHHeap, m::MCModel, s::MH, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x, m))

  while true
    iterate!(heap, m, s, r, t, produce)
  end
end

### Perform iteration for Metropolis-Hastings sampler

function iterate!(heap::MHHeap, m::MCModel, s::MH, r::MCRunner, t::MCTuner, send::Function)
  if t.verbose
    heap.tune.proposed += 1
  end

  heap.instate.successive = MCBaseSample(s.randproposal(heap.instate.current.sample))
  logtarget!(heap.instate.successive, m.eval)

  if s.symmetric
    heap.ratio = heap.instate.successive.logtarget-heap.instate.current.logtarget
  else
    heap.ratio = (heap.instate.successive.logtarget
      +s.logproposal(heap.instate.successive.sample, heap.instate.current.sample)
      -heap.instate.current.logtarget
      -s.logproposal(heap.instate.current.sample, heap.instate.successive.sample)
    )
  end
  if heap.ratio > 0 || (heap.ratio > log(rand()))
    heap.outstate = MCState(heap.instate.successive, heap.instate.current, Dict{Any, Any}("accept" => true))
    heap.instate.current = deepcopy(heap.instate.successive)

    if t.verbose
      heap.tune.accepted += 1
    end
  else
    heap.outstate = MCState(heap.instate.current, heap.instate.current, Dict{Any, Any}("accept" => false))
  end

  if t.verbose && heap.count <= r.burnin && mod(heap.count, t.period) == 0
    rate!(heap.tune)
    println("Burnin iteration $(heap.count) of $(r.burnin): ", round(100*heap.tune.rate, 2), " % acceptance rate")
  end

  heap.count += 1

  send(heap.outstate)
end
