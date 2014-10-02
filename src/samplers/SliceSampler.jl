### Slice sampler

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

### SliceSamplerStash type holds the internal state ("local variables") of the MALA sampler

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
