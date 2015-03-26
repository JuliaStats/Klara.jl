### MALA holds the fields of the MALA sampler
### These fields represent the initial user-defined state of the sampler
### These sampler fields are copied to the corresponding heap type fields, where the latter can be tuned

immutable MALA <: LMCSampler
  driftstep::Float64

  function MALA(ds::Float64)
    @assert ds > 0 "Drift step is not positive."
    new(ds)
  end
end

MALA(; driftstep::Float64=1.) = MALA(driftstep)

### MALAHeap type holds the internal state ("local variables") of the MALA sampler

type MALAHeap <: MCHeap{MCGradSample}
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

MALAHeap() =
  MALAHeap(MCState(MCGradSample(), MCGradSample()), MCState(MCGradSample(), MCGradSample()), VanillaMCTune(), 0, NaN,
  Float64[], NaN, NaN, NaN)

MALAHeap(l::Int, t::MCTune=VanillaMCTune()) =
  MALAHeap(MCState(MCGradSample(l), MCGradSample(l)), MCState(MCGradSample(l), MCGradSample(l)), t, 0, NaN,
    fill(NaN, l), NaN, NaN, NaN)

### Initialize MALA sampler

function initialize_heap(m::MCModel, s::MALA, r::MCRunner, t::MCTuner)
  @assert hasgradient(m) "MALA sampler requires model with gradient function."
  heap::MALAHeap = MALAHeap(m.size)

  heap.instate.current = MCGradSample(copy(m.init))
  gradlogtargetall!(heap.instate.current, m.evalallg)
  @assert isfinite(heap.instate.current.logtarget) "Initial values out of model support."

  if isa(t, VanillaMCTuner)
    heap.tune = VanillaMCTune()
  elseif isa(t, EmpiricalMCTuner)
    heap.tune = EmpiricalMCTune(s.driftstep)
  end

  heap.count = 1

  heap
end

function reset!(heap::MALAHeap, x::Vector{Float64}, m::MCModel)
  heap.instate.current = MCGradSample(copy(x))
  gradlogtargetall!(heap.instate.current, m.evalallg)
end

function initialize_task!(heap::MALAHeap, m::MCModel, s::MALA, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x, m))

  while true
    iterate!(heap, m, s, r, t, produce)
  end
end

### Perform iteration for MALA sampler

function iterate!(heap::MALAHeap, m::MCModel, s::MALA, r::MCRunner, t::MCTuner, send::Function)
  if isa(t, VanillaMCTuner)
    heap.driftstep = s.driftstep
    if t.verbose
      heap.tune.proposed += 1
    end
  elseif isa(t, EmpiricalMCTuner)
    heap.driftstep = heap.tune.step
    heap.tune.proposed += 1
  end

  heap.smean = heap.instate.current.sample+(heap.driftstep/2.)*heap.instate.current.gradlogtarget
  heap.instate.successive = MCGradSample(heap.smean+sqrt(heap.driftstep)*randn(m.size))
  gradlogtargetall!(heap.instate.successive, m.evalallg)
  heap.pnewgivenold =
    sum(-(heap.smean-heap.instate.successive.sample).^2/(2*heap.driftstep).-log(2*pi*heap.driftstep)/2)

  heap.smean = heap.instate.successive.sample+(heap.driftstep/2)*heap.instate.successive.gradlogtarget
  heap.poldgivennew =
    sum(-(heap.smean-heap.instate.current.sample).^2/(2*heap.driftstep).-log(2*pi*heap.driftstep)/2)

  heap.ratio = heap.instate.successive.logtarget+heap.poldgivennew-heap.instate.current.logtarget-heap.pnewgivenold
  if heap.ratio > 0 || (heap.ratio > log(rand()))
    heap.outstate = MCState(heap.instate.successive, heap.instate.current, Dict{Any, Any}("accept" => true))
    heap.instate.current = deepcopy(heap.instate.successive)

    if isa(t, VanillaMCTuner) && t.verbose
      heap.tune.accepted += 1
    elseif isa(t, EmpiricalMCTuner)
      heap.tune.accepted += 1
    end
  else
    heap.outstate = MCState(heap.instate.current, heap.instate.current, Dict{Any, Any}("accept" => false))
  end

  if isa(t, VanillaMCTuner) && t.verbose && heap.count <= r.burnin && mod(heap.count, t.period) == 0
    rate!(heap.tune)
    println("Burnin iteration $(heap.count) of $(r.burnin): ", round(100*heap.tune.rate, 2), " % acceptance rate")
  elseif isa(t, EmpiricalMCTuner) && heap.count <= r.burnin && mod(heap.count, t.period) == 0
    adapt!(heap.tune, t)
    reset!(heap.tune)
    if t.verbose
      println("Burnin iteration $(heap.count) of $(r.burnin): ", round(100*heap.tune.rate, 2), " % acceptance rate")
    end
  end

  heap.count += 1

  send(heap.outstate)
end
