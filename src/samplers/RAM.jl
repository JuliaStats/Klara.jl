### RAM holds the fields of the robust adaptive Metropolis (RAM) sampler

immutable RAM <: MHSampler
  jumpscale::Float64 # Scale factor for adapting the jump size
  targetrate::Float64 # Target acceptance rate

  function RAM(s::Float64, r::Float64)
    @assert s > 0 "Scale factor for adapting the jump size is not positive."
    @assert 0 < r < 1 "Target acceptance rate is not between 0 and 1"
    new(s, r)
  end
end

RAM() = RAM(1., 0.234)
RAM(s::Float64) = RAM(s, 0.234)

RAM(; jumpscale::Float64=1.0, targetrate::Float64=0.234) = RAM(jumpscale, targetrate)

### RAMHeap type holds the internal state ("local variables") of the RAM sampler

type RAMHeap <: MCHeap{MCBaseSample}
  instate::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCBaseSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  ratio::Float64
  eta::Float64
  SS::Matrix{Float64}
  S::Matrix{Float64}
  rnormal::Vector{Float64}
end

RAMHeap() =
  RAMHeap(MCState(MCBaseSample(), MCBaseSample()), MCState(MCBaseSample(), MCBaseSample()), VanillaMCTune(), 0, NaN,
  NaN, Array(Float64, 0, 0), Array(Float64, 0, 0), Float64[])

RAMHeap(l::Int) =
  RAMHeap(MCState(MCBaseSample(l), MCBaseSample(l)), MCState(MCBaseSample(l), MCBaseSample(l)), VanillaMCTune(), 0,
  NaN, NaN, fill(NaN, l, l), fill(NaN, l, l), fill(NaN, l))

### Initialize RAM sampler

function initialize_heap(m::MCModel, s::RAM, r::MCRunner, t::MCTuner)
  heap::RAMHeap = RAMHeap(m.size)

  heap.instate.current = MCBaseSample(copy(m.init))
  logtarget!(heap.instate.current, m.eval)
  @assert isfinite(heap.instate.current.logtarget) "Initial values out of model support."

  heap.S = Float64[i == j ? (m.scale.*s.jumpscale)[i] : 0. for i in 1:m.size, j in 1:m.size]

  heap.tune = VanillaMCTune()

  heap.count = 1

  heap
end

function reset!(heap::RAMHeap, x::Vector{Float64}, m::MCModel)
  heap.instate.current = MCBaseSample(copy(x))
  logtarget!(heap.instate.current, m.eval)
end

function initialize_task!(heap::RAMHeap, m::MCModel, s::RAM, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x, m))

  while true
    iterate!(heap, m, s, r, t, produce)
  end
end

### Perform iteration for RAM sampler

function iterate!(heap::RAMHeap, m::MCModel, s::RAM, r::MCRunner, t::MCTuner, send::Function)
  if t.verbose
    heap.tune.proposed += 1
  end

  heap.rnormal = randn(m.size)
  heap.instate.successive = MCBaseSample(heap.instate.current.sample+heap.S*heap.rnormal)
  logtarget!(heap.instate.successive, m.eval)

  heap.ratio = heap.instate.successive.logtarget-heap.instate.current.logtarget
  if heap.ratio > 0 || (heap.ratio > log(rand()))
    heap.outstate = MCState(heap.instate.successive, heap.instate.current,
    {"accept" => true, "scale" => trace(heap.S)})
    heap.instate.current = deepcopy(heap.instate.successive)

    if t.verbose
      heap.tune.accepted += 1
    end
  else
    heap.outstate = MCState(heap.instate.current, heap.instate.current,
    {"accept" => false, "scale" => trace(heap.S)})
  end

  if t.verbose && heap.count <= r.burnin && mod(heap.count, t.period) == 0
    rate!(heap.tune)
    println("Burnin iteration $(heap.count) of $(r.burnin): ", round(100*heap.tune.rate, 2), " % acceptance rate")
  end

  heap.eta = min(1, m.size*heap.count^(-2/3))

  heap.SS =
    (heap.rnormal*heap.rnormal')/dot(heap.rnormal, heap.rnormal)*heap.eta*(min(1, exp(heap.ratio))-s.targetrate)
  heap.SS = heap.S*(eye(m.size)+heap.SS)*heap.S'
  heap.S = chol(heap.SS)'

  heap.count += 1

  send(heap.outstate)
end
