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

### SliceSamplerStash type holds the internal state ("local variables") of the slice sampler

type SliceSamplerStash <: MCStash{MCBaseSample}
  state::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  count::Int # Current number of iterations
  widths::Vector{Float64} # Step sizes for expanding the slice for the current Monte Carlo iteration 
  xl::Vector{Float64}
  xr::Vector{Float64}
  xprime::Vector{Float64}
  loguprime::Float64
  runiform::Float64
end

SliceSamplerStash() =
  SliceSamplerStash(MCState(MCBaseSample(), MCBaseSample()), 0, Float64[], Float64[], Float64[], Float64[], NaN, NaN)

SliceSamplerStash(l::Int) =
  SliceSamplerStash(MCState(MCBaseSample(l), MCBaseSample(l)), 0, fill(NaN, l), fill(NaN, l), fill(NaN, l), fill(NaN, l), NaN, NaN)

### Initialize slice sampler

function initialize_stash(m::MCModel, s::SliceSampler, r::MCRunner, t::MCTuner)
  stash::SliceSamplerStash = SliceSamplerStash(m.size)

  stash.state.successive = MCBaseSample(copy(m.init))
  logtarget!(stash.state.successive, m.eval)
  @assert isfinite(stash.state.successive.logtarget) "Initial values out of model support."

  if length(s.widths) == 0
    stash.widths = ones(m.size)
  else
    @assert length(s.widths) == m.size "Length of step sizes in widths must be equal to model size."
    stash.widths = s.widths
  end

  stash.count = 1

  stash
end

function reset!(stash::SliceSamplerStash, x::Vector{Float64})
  stash.state.successive = MCBaseSample(copy(x))
  logtarget!(stash.state.successive, m.eval)
end

function initialize_task!(stash::SliceSamplerStash, m::MCModel, s::SliceSampler, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(stash, x))

  while true
    iterate!(stash, m, s, r, t, produce)
  end
end

### Perform iteration for slice sampler

function iterate!(stash::SliceSamplerStash, m::MCModel, s::SliceSampler, r::MCRunner, t::MCTuner, send::Function)
  for j = 1:m.size
    stash.loguprime = log(rand())+stash.state.successive.logtarget
    stash.xl = copy(stash.state.successive.sample)
    stash.xr = copy(stash.state.successive.sample)
    stash.xprime = copy(stash.state.successive.sample)

    # Create a horizontal interval (stash.xl, stash.xr) enclosing xx
    stash.runiform = rand()
    stash.xl[j] = stash.state.successive.sample[j]-stash.runiform*stash.widths[j]
    stash.xr[j] = stash.state.successive.sample[j]+(1-stash.runiform)*stash.widths[j]
    if s.stepout
      while m.eval(stash.xl) > stash.loguprime
        stash.xl[j] -= stash.widths[j]
      end
      while m.eval(stash.xr) > stash.loguprime
        stash.xr[j] += stash.widths[j]
      end
    end

    # Inner loop: propose xprimes and shrink interval until good one is found
    while true
      stash.xprime[j] = rand()*(stash.xr[j]-stash.xl[j])+stash.xl[j]
      stash.state.successive.logtarget = m.eval(stash.xprime)
      if stash.state.successive.logtarget > stash.loguprime
        break
      else
        if (stash.xprime[j] > stash.state.successive.sample[j])
          stash.xr[j] = stash.xprime[j]
        elseif (stash.xprime[j] < stash.state.successive.sample[j])
          stash.xl[j] = stash.xprime[j]
        else
          @assert false "Shrunk to current position and still not acceptable."
        end
      end
    end

    stash.state.successive.sample[j] = stash.xprime[j]
  end

  stash.count += 1

  send(MCState(stash.state.successive, MCBaseSample(), Dict()))
end
