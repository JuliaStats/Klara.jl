### Abstract SMMALA state

abstract SMMALAState <: MCSamplerState

### SMMALA state subtypes

## UnvSMMALAState holds the internal state ("local variables") of the SMMALA sampler for univariate parameters

type UnvSMMALAState <: SMMALAState
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by SMMALA
  tune::MCTunerState
  ratio::Real
  μ::Real
  newinvtensor::Real
  oldinvtensor::Real
  cholinvtensor::Real
  newfirstterm::Real
  oldfirstterm::Real

  function UnvSMMALAState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    μ::Real,
    newinvtensor::Real,
    oldinvtensor::Real,
    cholinvtensor::Real,
    newfirstterm::Real,
    oldfirstterm::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, μ, newinvtensor, oldinvtensor, cholinvtensor, newfirstterm, oldfirstterm)
  end
end

UnvSMMALAState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune()) =
  UnvSMMALAState(pstate, tune, NaN, NaN, NaN, NaN, NaN, NaN, NaN)

## MuvSMMALAState holds the internal state ("local variables") of the SMMALA sampler for multivariate parameters

type MuvSMMALAState <: SMMALAState
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by SMMALA
  tune::MCTunerState
  ratio::Real
  μ::RealVector
  newinvtensor::RealMatrix
  oldinvtensor::RealMatrix
  cholinvtensor::RealLowerTriangular
  newfirstterm::RealVector
  oldfirstterm::RealVector

  function MuvSMMALAState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    μ::RealVector,
    newinvtensor::RealMatrix,
    oldinvtensor::RealMatrix,
    cholinvtensor::RealLowerTriangular,
    newfirstterm::RealVector,
    oldfirstterm::RealVector
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, μ, newinvtensor, oldinvtensor, cholinvtensor, newfirstterm, oldfirstterm)
  end
end

MuvSMMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvSMMALAState(
    pstate,
    tune,
    NaN,
    Array(eltype(pstate), pstate.size),
    Array(eltype(pstate), pstate.size, pstate.size),
    Array(eltype(pstate), pstate.size, pstate.size),
    RealLowerTriangular(Array(eltype(pstate), pstate.size, pstate.size)),
    Array(eltype(pstate), pstate.size),
    Array(eltype(pstate), pstate.size)
  )

### Metropolis-adjusted Langevin Algorithm (SMMALA)

immutable SMMALA <: LMCSampler
  driftstep::Real

  function SMMALA(driftstep::Real)
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep)
  end
end

SMMALA() = SMMALA(1.)

### Initialize SMMALA sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::SMMALA
)
  parameter.uptotensorlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of parameter support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of parameter support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of parameter support"
end

## Initialize SMMALA state

function sampler_state(
  sampler::SMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
)
  sstate = UnvSMMALAState(generate_empty(pstate), tuner_state(sampler, tuner))
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate
end

function sampler_state(
  sampler::SMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvSMMALAState(generate_empty(pstate), tuner_state(sampler, tuner))
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate
end

### Reset SMMALA sampler

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::SMMALA
)
  pstate.value = x
  parameter.uptotensorlogtarget!(pstate)
end

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::SMMALA
)
  pstate.value = copy(x)
  parameter.uptotensorlogtarget!(pstate)
end

## Reset sampler state

function reset!{F<:VariateForm}(
  sstate::SMMALAState,
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::SMMALA,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
end

Base.show(io::IO, sampler::SMMALA) = print(io, "SMMALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::SMMALA) = show(io, sampler)
