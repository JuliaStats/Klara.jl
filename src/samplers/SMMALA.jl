### SMMALA holds the fields of the SMMALA sampler
### These fields represent the initial user-defined state of the sampler
### These sampler fields are copied to the corresponding heap type fields, where the latter can be tuned

immutable SMMALA <: LMCSampler
  driftstep::Float64

  function SMMALA(ds::Float64)
    @assert ds > 0 "Drift step is not positive."
    new(ds)
  end
end

SMMALA(; driftstep::Float64=1.) = SMMALA(driftstep)

### SMMALAHeap type holds the internal state ("local variables") of the SMMALA sampler

type SMMALAHeap <: MCHeap{MCTensorSample}
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
  cholinvtensor::Matrix{Float64}
end

SMMALAHeap() =
  SMMALAHeap(MCState(MCTensorSample(), MCTensorSample()), MCState(MCTensorSample(), MCTensorSample()), VanillaMCTune(),
  0, NaN, Float64[], NaN, NaN, NaN, Array(Float64, 0, 0), Array(Float64, 0, 0), Float64[], Float64[],
  Array(Float64, 0, 0))

SMMALAHeap(l::Int, t::MCTune=VanillaMCTune()) =
  SMMALAHeap(MCState(MCTensorSample(l), MCTensorSample(l)), MCState(MCTensorSample(l), MCTensorSample(l)), t, 0, NaN,
    fill(NaN, l), NaN, NaN, NaN, fill(NaN, l, l), fill(NaN, l, l), fill(NaN, l), fill(NaN, l),
    Array(Float64, 0, 0))

### Initialize SMMALA sampler

function initialize_heap(m::MCModel, s::SMMALA, r::MCRunner, t::MCTuner)
  @assert hasgradient(m) "SMMALA sampler requires model with gradient function."
  @assert hastensor(m) "SMMALA sampler requires model with tensor function."
  heap::SMMALAHeap = SMMALAHeap(m.size)

  heap.instate.current = MCTensorSample(copy(m.init))
  tensorlogtargetall!(heap.instate.current, m.evalallt)
  @assert isfinite(heap.instate.current.logtarget) "Initial values out of model support."

  heap.current_invtensor = inv(heap.instate.current.tensorlogtarget)
  heap.current_termone = heap.current_invtensor*heap.instate.current.gradlogtarget

  if isa(t, VanillaMCTuner)
    heap.tune = VanillaMCTune()
  elseif isa(t, EmpiricalMCTuner)
    heap.tune = EmpiricalMCTune(s.driftstep)
  end

  heap.count = 1

  heap
end

function reset!(heap::SMMALAHeap, x::Vector{Float64})
  heap.instate.current = MCTensorSample(copy(x))
  tensorlogtargetall!(heap.instate.current, m.evalallt)
end

function initialize_task!(heap::SMMALAHeap, m::MCModel, s::SMMALA, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x))

  while true
    iterate!(heap, m, s, r, t, produce)
  end
end

### Perform iteration for SMMALA sampler

function iterate!(heap::SMMALAHeap, m::MCModel, s::SMMALA, r::MCRunner, t::MCTuner, send::Function)
  if isa(t, VanillaMCTuner)
    heap.driftstep = s.driftstep
    if t.verbose
      heap.tune.proposed += 1
    end
  elseif isa(t, EmpiricalMCTuner)
    heap.driftstep = heap.tune.step
    heap.tune.proposed += 1
  end

  # Calculate the drift term
  heap.smean = heap.instate.current.sample+(heap.driftstep/2)*heap.current_termone

  # Calculate proposed parameters
  heap.cholinvtensor = chol(heap.driftstep*heap.current_invtensor)
  heap.instate.successive = MCTensorSample(heap.smean+heap.cholinvtensor'*randn(m.size))

  # Update model based on the proposed parameters
  tensorlogtargetall!(heap.instate.successive, m.evalallt)

  heap.pnewgivenold = (-sum(log(diag(heap.cholinvtensor)))
    -(0.5*(heap.smean-heap.instate.successive.sample)'*(heap.instate.current.tensorlogtarget/heap.driftstep)
    *(heap.smean-heap.instate.successive.sample))[1])

  heap.successive_invtensor = inv(heap.instate.successive.tensorlogtarget)
  heap.successive_termone = heap.successive_invtensor*heap.instate.successive.gradlogtarget

  # Calculate the drift term
  heap.smean = heap.instate.successive.sample+(heap.driftstep/2)*heap.successive_termone

  heap.poldgivennew = (-sum(log(diag(chol(heap.driftstep*eye(m.size)*heap.successive_invtensor))))
    -(0.5*(heap.smean-heap.instate.current.sample)'*(heap.instate.successive.tensorlogtarget/heap.driftstep)
    *(heap.smean-heap.instate.current.sample))[1])

  heap.ratio = heap.instate.successive.logtarget+heap.poldgivennew-heap.instate.current.logtarget-heap.pnewgivenold

  if heap.ratio > 0 || (heap.ratio > log(rand()))
    heap.outstate = MCState(heap.instate.successive, heap.instate.current, {"accept" => true})
    heap.instate.current = deepcopy(heap.instate.successive)
    heap.current_invtensor, heap.current_termone = copy(heap.successive_invtensor), copy(heap.successive_termone)

    if isa(t, VanillaMCTuner) && t.verbose
      heap.tune.accepted += 1
    elseif isa(t, EmpiricalMCTuner)
      heap.tune.accepted += 1
    end
  else
    heap.outstate = MCState(heap.instate.current, heap.instate.current, {"accept" => false})
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
