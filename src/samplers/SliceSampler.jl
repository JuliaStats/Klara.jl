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
