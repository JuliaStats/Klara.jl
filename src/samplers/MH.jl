### MHState

# MHState holds the internal state ("local variables") of the Metropolis-Hastings sampler

type MHState{S<:ValueSupport, F<:VariateForm} <: MHSamplerState{F}
  proposal::Distribution{F, S} # Proposal distribution
  pstate::ParameterState{S, F} # Parameter state used internally by MH
  tune::MCTunerState
  ratio::Real # Acceptance ratio

  function MHState(proposal::Distribution{F, S}, pstate::ParameterState{S, F}, tune::MCTunerState, ratio::Real)
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(proposal, pstate, tune, ratio)
  end
end

MHState{S<:ValueSupport, F<:VariateForm}(
  proposal::Distribution{F, S},
  pstate::ParameterState{S, F},
  tune::MCTunerState,
  ratio::Real
) =
  MHState{S, F}(proposal, pstate, tune, ratio)

MHState{S<:ValueSupport, F<:VariateForm}(
  proposal::Distribution{F, S},
  pstate::ParameterState{S, F},
  tune::MCTunerState=BasicMCTune()
) =
  MHState(proposal, pstate, tune, NaN)

### Metropolis-Hastings (MH) sampler

# In its most general case it accommodates an asymmetric proposal distribution
# For symmetric proposals, the proposal correction factor equals 1, so the logproposal field is set to nothing
# For non-normalised proposals, the lognormalise() method is used for calculating the proposal correction factor

immutable MH <: MHSampler
  symmetric::Bool # If symmetric=true then the proposal distribution is symmetric, else it is asymmetric
  normalised::Bool # If normalised=true then the proposal distribution is normalised, else it is non-normalised
  setproposal::Function # Function for setting the proposal distribution
end

MH(setproposal::Function; signature::Symbol=:high, args...) = MH(setproposal, Val{signature}; args...)

MH(setproposal::Function, ::Type{Val{:low}}; symmetric::Bool=true, normalised::Bool=true) =
  MH(symmetric, normalised, setproposal)

MH(
  setproposal::Function,
  ::Type{Val{:high}};
  symmetric::Bool=true,
  normalised::Bool=true
) =
  MH(symmetric, normalised, eval(codegen_lowlevel_variable_method(setproposal, nothing, false, Symbol[], 0)))

# Random-walk Metropolis, i.e. Metropolis with a normal proposal distribution

MH{N<:Real}(σ::Matrix{N}) = MH(x::Vector{N} -> MvNormal(x, σ), signature=:high)
MH{N<:Real}(σ::Vector{N}) = MH(x::Vector{N} -> MvNormal(x, σ), signature=:high)
MH{N<:Real}(σ::N) = MH(x::N -> Normal(x, σ), signature=:high)
MH{N<:Real}(::Type{N}=Float64) = MH(x::N -> Normal(x, 1.0), signature=:high)

### Initialize Metropolis-Hastings sampler

## Initialize parameter state

function initialize!{S<:ValueSupport, F<:VariateForm}(pstate::ParameterState{S, F}, parameter::Parameter{S, F}, sampler::MH)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
end

## Initialize MHState

sampler_state{S<:ValueSupport, F<:VariateForm}(
  parameter::Parameter{S, F},
  sampler::MH,
  tuner::MCTuner,
  pstate::ParameterState{S, F},
  vstate::VariableStateVector
) =
  MHState(sampler.setproposal(pstate), generate_empty(pstate), tuner_state(parameter, sampler, tuner))

## Reset parameter state

function reset!(pstate::UnivariateParameterState, x::Real, parameter::UnivariateParameter, sampler::MH)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!(pstate::MultivariateParameterState, x::RealVector, parameter::MultivariateParameter, sampler::MH)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

reset!{S<:ValueSupport, F<:VariateForm}(
  sstate::MHState{S, F},
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MCSampler,
  tuner::MCTuner
) =
  reset!(sstate.tune, sampler, tuner)

function Base.show(io::IO, sampler::MH)
  issymmetric = sampler.symmetric ? "symmetric" : "non-symmmetric"
  isnormalised = sampler.normalised ? "normalised" : "non-normalised"
  print(io, "MH sampler: $issymmetric $isnormalised proposal")
end

Base.writemime(io::IO, ::MIME"text/plain", sampler::MH) = show(io, sampler)
