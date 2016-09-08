### Abstract MALA state

abstract MALAState{F<:VariateForm} <: LMCSamplerState{F}

### MALA state subtypes

## UnvMALAState holds the internal state ("local variables") of the MALA sampler for univariate parameters

type UnvMALAState <: MALAState{Univariate}
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by MALA
  tune::MCTunerState
  ratio::Real
  μ::Real

  function UnvMALAState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState, ratio::Real, μ::Real)
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(pstate, tune, ratio, μ)
  end
end

UnvMALAState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune()) =
  UnvMALAState(pstate, tune, NaN, NaN)

## MuvMALAState holds the internal state ("local variables") of the MALA sampler for multivariate parameters

type MuvMALAState <: MALAState{Multivariate}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by MALA
  tune::MCTunerState
  ratio::Real
  μ::RealVector

  function MuvMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState, ratio::Real, μ::RealVector)
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(pstate, tune, ratio, μ)
  end
end

MuvMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvMALAState(pstate, tune, NaN, Array(eltype(pstate), pstate.size))

### Metropolis-adjusted Langevin Algorithm (MALA)

immutable MALA <: LMCSampler
  driftstep::Real

  function MALA(driftstep::Real)
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep)
  end
end

MALA() = MALA(1.)

### Initialize MALA sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::MALA
)
  parameter.uptogradlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
end

## Initialize MALA state

sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
) =
  UnvMALAState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))

sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
) =
  MuvMALAState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA
)
  pstate.value = x
  parameter.uptogradlogtarget!(pstate)
end

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA
)
  pstate.value = copy(x)
  parameter.uptogradlogtarget!(pstate)
end

Base.show(io::IO, sampler::MALA) = print(io, "MALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::MALA) = show(io, sampler)
