### SliceSampler holds the fields of the slice sampler

immutable SliceSampler <: MCSampler
  widths::Vector{Float64} # Step sizes for initially expanding the slice
  stepout::Bool # Protects against the case of passing in small widths

  function SliceSampler(widths::Vector{Float64}, stepout::Bool)
    for x = widths
    @assert x > 0 "Widths should be positive."
    end
    new(widths, stepout)
  end
end

SliceSampler(widths::Vector{Float64}) = SliceSampler(widths, true)
SliceSampler(stepout::Bool) = SliceSampler(Float64[], stepout)

SliceSampler(; widths::Vector{Float64}=Float64[], stepout::Bool=true) = SliceSampler(widths, stepout)

### SliceSamplerHeap type holds the internal state ("local variables") of the slice sampler

type SliceSamplerHeap <: MCHeap{MCBaseSample}
  state::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  count::Int # Current number of iterations
  widths::Vector{Float64} # Step sizes for expanding the slice for the current Monte Carlo iteration 
  xl::Vector{Float64}
  xr::Vector{Float64}
  xprime::Vector{Float64}
  loguprime::Float64
  runiform::Float64
end

SliceSamplerHeap() =
  SliceSamplerHeap(MCState(MCBaseSample(), MCBaseSample()), 0, Float64[], Float64[], Float64[], Float64[], NaN, NaN)

SliceSamplerHeap(l::Int) =
  SliceSamplerHeap(MCState(MCBaseSample(l), MCBaseSample(l)), 0, fill(NaN, l), fill(NaN, l), fill(NaN, l), fill(NaN, l), NaN, NaN)

### Initialize slice sampler

function initialize_heap(m::MCModel, s::SliceSampler, r::MCRunner, t::MCTuner)
  heap::SliceSamplerHeap = SliceSamplerHeap(m.size)

  heap.state.successive = MCBaseSample(copy(m.init))
  logtarget!(heap.state.successive, m.eval)
  @assert isfinite(heap.state.successive.logtarget) "Initial values out of model support."

  if length(s.widths) == 0
    heap.widths = ones(m.size)
  else
    @assert length(s.widths) == m.size "Length of step sizes in widths must be equal to model size."
    heap.widths = s.widths
  end

  heap.count = 1

  heap
end

function reset!(heap::SliceSamplerHeap, x::Vector{Float64})
  heap.state.successive = MCBaseSample(copy(x))
  logtarget!(heap.state.successive, m.eval)
end

function initialize_task!(heap::SliceSamplerHeap, m::MCModel, s::SliceSampler, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x))

  while true
    iterate!(heap, m, s, r, t, produce)
  end
end

### Perform iteration for slice sampler

function iterate!(heap::SliceSamplerHeap, m::MCModel, s::SliceSampler, r::MCRunner, t::MCTuner, send::Function)
  for j = 1:m.size
    heap.loguprime = log(rand())+heap.state.successive.logtarget
    heap.xl = copy(heap.state.successive.sample)
    heap.xr = copy(heap.state.successive.sample)
    heap.xprime = copy(heap.state.successive.sample)

    # Create a horizontal interval (heap.xl, heap.xr) enclosing xx
    heap.runiform = rand()
    heap.xl[j] = heap.state.successive.sample[j]-heap.runiform*heap.widths[j]
    heap.xr[j] = heap.state.successive.sample[j]+(1-heap.runiform)*heap.widths[j]
    if s.stepout
      while m.eval(heap.xl) > heap.loguprime
        heap.xl[j] -= heap.widths[j]
      end
      while m.eval(heap.xr) > heap.loguprime
        heap.xr[j] += heap.widths[j]
      end
    end

    # Inner loop: propose xprimes and shrink interval until good one is found
    while true
      heap.xprime[j] = rand()*(heap.xr[j]-heap.xl[j])+heap.xl[j]
      heap.state.successive.logtarget = m.eval(heap.xprime)
      if heap.state.successive.logtarget > heap.loguprime
        break
      else
        if (heap.xprime[j] > heap.state.successive.sample[j])
          heap.xr[j] = heap.xprime[j]
        elseif (heap.xprime[j] < heap.state.successive.sample[j])
          heap.xl[j] = heap.xprime[j]
        else
          @assert false "Shrunk to current position and still not acceptable."
        end
      end
    end

    heap.state.successive.sample[j] = heap.xprime[j]
  end

  heap.count += 1

  send(MCState(heap.state.successive, MCBaseSample(), Dict()))
end
