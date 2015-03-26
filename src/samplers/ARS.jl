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

### ARSHeap type holds the internal state ("local variables") of the ARS sampler

type ARSHeap <: MCHeap{MCBaseSample}
  instate::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCBaseSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  jumpscale::Vector{Float64}
  logproposal::Float64
  weight::Float64
end

ARSHeap() =
  ARSHeap(MCState(MCBaseSample(), MCBaseSample()), MCState(MCBaseSample(), MCBaseSample()), VanillaMCTune(), 0,
  Float64[], NaN, NaN)

ARSHeap(l::Int) =
  ARSHeap(MCState(MCBaseSample(l), MCBaseSample(l)), MCState(MCBaseSample(l), MCBaseSample(l)), VanillaMCTune(), 0,
  Float64[], NaN, NaN)

### Initialize Metropolis-Hastings sampler

function initialize_heap(m::MCModel, s::ARS, r::MCRunner, t::MCTuner)
  heap::ARSHeap = ARSHeap(m.size)

  heap.instate.current = MCBaseSample(copy(m.init))
  logtarget!(heap.instate.current, m.eval)
  @assert isfinite(heap.instate.current.logtarget) "Initial values out of model support."

  heap.jumpscale = m.scale.*s.jumpscale  # rescale model scale by sampler scale

  heap.tune = VanillaMCTune()

  heap.count = 1

  heap
end

function reset!(heap::ARSHeap, x::Vector{Float64}, m::MCModel)
  heap.instate.current = MCBaseSample(copy(x))
  logtarget!(heap.instate.current, m.eval)
end

function initialize_task!(heap::ARSHeap, m::MCModel, s::ARS, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x))

  while true
    iterate!(heap, m, s, r, t, produce)
  end
end

### Perform iteration for ARS sampler

function iterate!(heap::ARSHeap, m::MCModel, s::ARS, r::MCRunner, t::MCTuner, send::Function)
  if t.verbose
    heap.tune.proposed += 1
  end

  heap.instate.successive = MCBaseSample(heap.instate.current.sample+randn(m.size).*heap.jumpscale)
  logtarget!(heap.instate.successive, m.eval)
  heap.logproposal = s.logproposal(heap.instate.successive.sample)

  heap.weight = heap.instate.successive.logtarget.-s.proposalscale.-heap.logproposal
  if heap.weight > log(rand())
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
