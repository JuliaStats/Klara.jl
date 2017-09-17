### MHState

# MHState holds the internal state ("local variables") of the Metropolis-Hastings sampler

mutable struct MHState{S<:ValueSupport, F<:VariateForm} <: MHSamplerState{F}
  proposal::Distribution{F, S} # Proposal distribution
  pstate::ParameterState{S, F} # Parameter state used internally by MH
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  diagnosticindices::Dict{Symbol, Integer}

  function MHState{S, F}(
    proposal::Distribution{F, S},
    pstate::ParameterState{S, F},
    tune::MCTunerState,
    ratio::Real,
    diagnosticindices::Dict{Symbol, Integer}
  ) where {S<:ValueSupport, F<:VariateForm}
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(proposal, pstate, tune, ratio, diagnosticindices)
  end
end

MHState(
  proposal::Distribution{F, S},
  pstate::ParameterState{S, F},
  tune::MCTunerState,
  ratio::Real
) where {S<:ValueSupport, F<:VariateForm} =
  MHState{S, F}(proposal, pstate, tune, ratio, Dict{Symbol, Integer}())

MHState(
  proposal::Distribution{F, S},
  pstate::ParameterState{S, F},
  tune::MCTunerState=BasicMCTune()
) where {S<:ValueSupport, F<:VariateForm} =
  MHState(proposal, pstate, tune, NaN)

### Metropolis-Hastings (MH) sampler

# In its most general case it accommodates an asymmetric proposal distribution
# For symmetric proposals, the proposal correction factor equals 1, so the logproposal field is set to nothing
# For non-normalised proposals, the lognormalise() method is used for calculating the proposal correction factor

struct MH <: MHSampler
  symmetric::Bool # If symmetric=true then the proposal distribution is symmetric, else it is asymmetric
  normalised::Bool # If normalised=true then the proposal distribution is normalised, else it is non-normalised
  setproposal::Function # Function for setting the proposal distribution
end

MH(setproposal::Function; signature::Symbol=:high, args...) = MH(setproposal, Val{signature}; args...)

MH(setproposal::Function, ::Type{Val{:low}}; symmetric::Bool=true, normalised::Bool=true) =
  MH(symmetric, normalised, setproposal)

MH(setproposal::Function, ::Type{Val{:high}}; symmetric::Bool=true, normalised::Bool=true) =
  MH(symmetric, normalised, _state -> setproposal(_state.value))

# Random-walk Metropolis, i.e. Metropolis with a normal proposal distribution

MH(σ::Matrix{N}) where {N<:Real} = MH(x::Vector{N} -> MvNormal(x, σ), signature=:high)
MH(σ::Vector{N}) where {N<:Real} = MH(x::Vector{N} -> MvNormal(x, σ), signature=:high)
MH(σ::N) where {N<:Real} = MH(x::N -> Normal(x, σ), signature=:high)
MH(::Type{N}=Float64) where {N<:Real} = MH(x::N -> Normal(x, 1.0), signature=:high)

### Initialize Metropolis-Hastings sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MH,
  outopts::Dict
) where {S<:ValueSupport, F<:VariateForm}
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

## Initialize MHState

function sampler_state(
  parameter::Parameter{S, F},
  sampler::MH,
  tuner::MCTuner,
  pstate::ParameterState{S, F},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
) where {S<:ValueSupport, F<:VariateForm}
  sstate = MHState(sampler.setproposal(pstate), generate_empty(pstate), tuner_state(parameter, sampler, tuner))
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

## Reset parameter state

function reset!(pstate::UnivariateParameterState, x::Real, parameter::UnivariateParameter, sampler::MH)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!(pstate::MultivariateParameterState, x::RealVector, parameter::MultivariateParameter, sampler::MH)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

reset!(
  sstate::MHState{S, F},
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MCSampler,
  tuner::MCTuner
) where {S<:ValueSupport, F<:VariateForm} =
  reset!(sstate.tune, sampler, tuner)

function show(io::IO, sampler::MH)
  issymmetric = sampler.symmetric ? "symmetric" : "non-symmmetric"
  isnormalised = sampler.normalised ? "normalised" : "non-normalised"
  print(io, "MH sampler: $issymmetric $isnormalised proposal")
end
