## MuvSliceSamplerState holds the internal state ("local variables") of the slice sampler for multivariate parameters

abstract SliceSamplerState{F<:VariateForm} <: MCSamplerState{F}

type MuvSliceSamplerState <: SliceSamplerState{Multivariate}
  xl::RealVector
  xr::RealVector
  xprime::RealVector
  loguprime::Real
  runiform::Real
end

MuvSliceSamplerState(s::Integer, t::Type=Float64) = MuvSliceSamplerState(Array(t, n), Array(t, n), Array(t, n), NaN, NaN)

### Slice sampler

immutable SliceSampler <: MCSampler
  widths::RealVector # Step sizes for initially expanding the slice
  stepout::Bool # Protects against the case of passing in small widths

  function SliceSampler(widths::RealVector, stepout::Bool)
    @assert all(i -> i > 0, widths) "All widths must be positive"
    new(widths, stepout)
  end
end

SliceSampler(widths::RealVector) = SliceSampler(widths, true)

SliceSampler(widths::Real=1., n::Integer=1, stepout::Bool=true) = SliceSampler(fill(widths, n), stepout)

### Initialize slice sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::SliceSampler,
  outopts::Dict
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
end

## Initialize SliceSampler state

sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::SliceSampler,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
) =
  MuvSliceSamplerState(pstate.size, eltype(pstate))
