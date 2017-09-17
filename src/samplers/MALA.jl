### Abstract MALA state

abstract type MALAState{F<:VariateForm} <: LMCSamplerState{F} end

### MALA state subtypes

## UnvMALAState holds the internal state ("local variables") of the MALA sampler for univariate parameters

mutable struct UnvMALAState <: MALAState{Univariate}
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by MALA
  tune::MCTunerState
  ratio::Real
  μ::Real
  diagnosticindices::Dict{Symbol, Integer}

  function UnvMALAState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    μ::Real,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(pstate, tune, ratio, μ, diagnosticindices)
  end
end

UnvMALAState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune()) =
  UnvMALAState(pstate, tune, NaN, NaN, Dict{Symbol, Integer}())

## MuvMALAState holds the internal state ("local variables") of the MALA sampler for multivariate parameters

mutable struct MuvMALAState <: MALAState{Multivariate}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by MALA
  tune::MCTunerState
  ratio::Real
  μ::RealVector
  diagnosticindices::Dict{Symbol, Integer}

  function MuvMALAState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    μ::RealVector,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(pstate, tune, ratio, μ, diagnosticindices)
  end
end

MuvMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvMALAState(pstate, tune, NaN, Array{eltype(pstate)}(pstate.size), Dict{Symbol, Integer}())

### Metropolis-adjusted Langevin Algorithm (MALA)

struct MALA <: LMCSampler
  driftstep::Real

  function MALA(driftstep::Real)
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep)
  end
end

MALA() = MALA(1.)

### Initialize MALA sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::MALA,
  outopts::Dict
) where F<:VariateForm
  parameter.uptogradlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite.(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

## Initialize MALA state

# Stopped at this point, fixing the sampler_state functions

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvMALAState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvMALAState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

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

show(io::IO, sampler::MALA) = print(io, "MALA sampler: drift step = $(sampler.driftstep)")
