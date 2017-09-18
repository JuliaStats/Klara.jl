### Abstract SMMALA state

abstract type SMMALAState{F<:VariateForm} <: LMCSamplerState{F} end

### SMMALA state subtypes

## UnvSMMALAState holds the internal state ("local variables") of the SMMALA sampler for univariate parameters

mutable struct UnvSMMALAState <: SMMALAState{Univariate}
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
  diagnosticindices::Dict{Symbol, Integer}

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
    oldfirstterm::Real,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(sqrttunestep)
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
    end
    new(
      pstate,
      tune,
      sqrttunestep,
      ratio,
      μ,
      newinvtensor,
      oldinvtensor,
      cholinvtensor,
      newfirstterm,
      oldfirstterm,
      diagnosticindices
    )
  end
end

UnvSMMALAState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune()) =
  UnvSMMALAState(pstate, tune, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, Dict{Symbol, Integer}())

## MuvSMMALAState holds the internal state ("local variables") of the SMMALA sampler for multivariate parameters

mutable struct MuvSMMALAState <: SMMALAState{Multivariate}
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
  diagnosticindices::Dict{Symbol, Integer}

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
    oldfirstterm::RealVector,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(sqrttunestep)
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
    end
    new(
      pstate,
      tune,
      sqrttunestep,
      ratio,
      μ,
      newinvtensor,
      oldinvtensor,
      cholinvtensor,
      newfirstterm,
      oldfirstterm,
      diagnosticindices
    )
  end
end

MuvSMMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvSMMALAState(
    pstate,
    tune,
    NaN,
    NaN,
    Array{eltype(pstate)}(pstate.size),
    Array{eltype(pstate)}(pstate.size, pstate.size),
    Array{eltype(pstate)}(pstate.size, pstate.size),
    RealLowerTriangular(Array{eltype(pstate)}(pstate.size, pstate.size)),
    Array{eltype(pstate)}(pstate.size),
    Array{eltype(pstate)}(pstate.size),
    Dict{Symbol, Integer}()
  )

### Metropolis-adjusted Langevin Algorithm (SMMALA)

struct SMMALA <: LMCSampler
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
  sampler::SMMALA,
  outopts::Dict
)
  parameter.uptotensorlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::SMMALA,
  outopts::Dict
)
  parameter.uptotensorlogtarget!(pstate)
  if sampler.transform != nothing
    pstate.tensorlogtarget[:, :] = sampler.transform(pstate.tensorlogtarget)
  end
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite.(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite.(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

## Initialize SMMALA state

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::SMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvSMMALAState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )

  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  println("tensorlogtarget = ", pstate.tensorlogtarget) 
  println("oldinvtensor = ", sstate.oldinvtensor)
  sstate.cholinvtensor = chol(sstate.oldinvtensor)
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget

  set_diagnosticindices!(sstate, [:accept], diagnostickeys)

  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::SMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvSMMALAState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )

  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor[:, :] = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(sstate.oldinvtensor)))
  sstate.oldfirstterm[:] = sstate.oldinvtensor*pstate.gradlogtarget

  set_diagnosticindices!(sstate, [:accept], diagnostickeys)

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
  sstate.cholinvtensor = chol(sstate.oldinvtensor)
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
  sstate.oldinvtensor[:, :] = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(sstate.oldinvtensor)))
  sstate.oldfirstterm[:] = sstate.oldinvtensor*pstate.gradlogtarget
end

show(io::IO, sampler::SMMALA) = print(io, "SMMALA sampler: drift step = $(sampler.driftstep)")
