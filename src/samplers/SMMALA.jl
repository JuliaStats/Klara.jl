### Abstract SMMALA state

abstract SMMALAState{F<:VariateForm} <: LMCSamplerState{F}

### SMMALA state subtypes

## UnvSMMALAState holds the internal state ("local variables") of the SMMALA sampler for univariate parameters

type UnvSMMALAState <: SMMALAState{Univariate}
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by SMMALA
  tune::MCTunerState
  sqrttunestep::Real
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
    sqrttunestep::Real,
    ratio::Real,
    μ::Real,
    newinvtensor::Real,
    oldinvtensor::Real,
    cholinvtensor::Real,
    newfirstterm::Real,
    oldfirstterm::Real
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(sqrttunestep)
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
    end
    new(pstate, tune, sqrttunestep, ratio, μ, newinvtensor, oldinvtensor, cholinvtensor, newfirstterm, oldfirstterm)
  end
end

UnvSMMALAState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune()) =
  UnvSMMALAState(pstate, tune, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)

## MuvSMMALAState holds the internal state ("local variables") of the SMMALA sampler for multivariate parameters

type MuvSMMALAState <: SMMALAState{Multivariate}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by SMMALA
  tune::MCTunerState
  sqrttunestep::Real
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
    sqrttunestep::Real,
    ratio::Real,
    μ::RealVector,
    newinvtensor::RealMatrix,
    oldinvtensor::RealMatrix,
    cholinvtensor::RealLowerTriangular,
    newfirstterm::RealVector,
    oldfirstterm::RealVector
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(sqrttunestep)
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
    end
    new(pstate, tune, sqrttunestep, ratio, μ, newinvtensor, oldinvtensor, cholinvtensor, newfirstterm, oldfirstterm)
  end
end

MuvSMMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvSMMALAState(
    pstate,
    tune,
    NaN,
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
  transform::Union{Function, Void} # Function for transforming metric tensor to a positive-definite matrix

  function SMMALA(driftstep::Real, transform::Union{Function, Void})
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep, transform)
  end
end

SMMALA(driftstep::Real=1.) = SMMALA(driftstep, nothing)

### Initialize SMMALA sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::SMMALA
)
  parameter.uptotensorlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"
end

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::SMMALA
)
  parameter.uptotensorlogtarget!(pstate)
  if sampler.transform != nothing
    pstate.tensorlogtarget = sampler.transform(pstate.tensorlogtarget)
  end
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"
end

## Initialize SMMALA state

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::SMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
)
  sstate = UnvSMMALAState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, :L)
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::SMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvSMMALAState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
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

function reset!(
  sstate::UnvSMMALAState,
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::SMMALA,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, :L)
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
end

function reset!(
  sstate::MuvSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::SMMALA,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
end

Base.show(io::IO, sampler::SMMALA) = print(io, "SMMALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::SMMALA) = show(io, sampler)
