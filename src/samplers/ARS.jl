### ARSState

# ARSState holds the internal state ("local variables") of the acceptance-rejection sampler

type ARSState{S<:ValueSupport, F<:VariateForm} <: MCSamplerState{F}
  pstate::ParameterState{S, F} # Parameter state used internally by ARS
  tune::MCTunerState
  logproposal::Real
  weight::Real
end

ARSState{S<:ValueSupport, F<:VariateForm}(
  pstate::ParameterState{S, F},
  tune::MCTunerState,
  logproposal::Real,
  weight::Real
) =
  ARSState{S, F}(pstate, tune, logproposal, weight)

ARSState{S<:ValueSupport, F<:VariateForm}(
  pstate::ParameterState{S, F},
  tune::MCTunerState=BasicMCTune()
) =
  ARSState(pstate, tune, NaN, NaN)

### Acceptance-rejection sampler (ARS)

immutable ARS <: MCSampler
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

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::ARS
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
end

## Initialize ARSState

sampler_state{S<:ValueSupport, F<:VariateForm}(
  parameter::Parameter{S, F},
  sampler::ARS,
  tuner::MCTuner,
  pstate::ParameterState{S, F},
  vstate::VariableStateVector
) =
  ARSState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))

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

reset!{S<:ValueSupport, F<:VariateForm}(
  sstate::ARSState{S, F},
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MCSampler,
  tuner::MCTuner
) =
  reset!(sstate.tune, sampler, tuner)

Base.show(io::IO, sampler::ARS) =
  print(io, "ARS sampler: proposal scale = $(sampler.proposalscale), jump scale = $(sampler.jumpscale)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::ARS) = show(io, sampler)
