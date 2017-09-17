### ARSState

# ARSState holds the internal state ("local variables") of the acceptance-rejection sampler

mutable struct ARSState{S<:ValueSupport, F<:VariateForm} <: MCSamplerState{F}
  pstate::ParameterState{S, F} # Parameter state used internally by ARS
  tune::MCTunerState
  logproposal::Real
  weight::Real
  diagnosticindices::Dict{Symbol, Integer}
end

ARSState(
  pstate::ParameterState{S, F},
  tune::MCTunerState=BasicMCTune()
) where {S<:ValueSupport, F<:VariateForm} =
  ARSState(pstate, tune, NaN, NaN, diagnosticindices)

### Acceptance-rejection sampler (ARS)

struct ARS <: MCSampler
  logproposal::Function # Possibly unnormalized log-proposal (envelope)
  proposalscale::Real # Scale factor for logproposal (envelope) to cover target
  jumpscale::Real # Scale factor for adapting jump size

  function ARS(logproposal::Function, proposalscale::Real, jumpscale::Real)
    @assert jumpscale > 0 "Scale factor for adapting jump size is not positive"
    new(logproposal, proposalscale, jumpscale)
  end
end

ARS(logproposal::Function; proposalscale::Real=1., jumpscale::Real=1.) = ARS(logproposal, proposalscale, jumpscale)

### Initialize ARS sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::ARS,
  outopts::Dict
) where F<:VariateForm
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array(Any, length(pstate.diagnostickeys))
  end
end

## Initialize ARSState

function sampler_state(
  parameter::Parameter{S, F},
  sampler::ARS,
  tuner::MCTuner,
  pstate::ParameterState{S, F},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
) where {S<:ValueSupport, F<:VariateForm}
  sstate = ARSState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::ARS
)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::ARS
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

reset!(
  sstate::ARSState{S, F},
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MCSampler,
  tuner::MCTuner
) where {S<:ValueSupport, F<:VariateForm} =
  reset!(sstate.tune, sampler, tuner)

show(io::IO, sampler::ARS) =
  print(io, "ARS sampler: proposal scale = $(sampler.proposalscale), jump scale = $(sampler.jumpscale)")
