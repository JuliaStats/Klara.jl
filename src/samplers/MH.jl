### MHState

# MHState holds the internal state ("local variables") of the Metropolis-Hastings sampler

type MHState <: MCSamplerState
  pstate::ParameterState # Parameter state used internally by MH
  tune::MCTunerState
  ratio::Real # Acceptance ratio

  function MHState(pstate::ParameterState, tune::MCTunerState, ratio::Real)
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio)
  end
end

MHState(pstate::ParameterState, tune::MCTunerState=VanillaMCTune()) = MHState(pstate, tune, NaN)

### Metropolis-Hastings (MH) sampler

# In its most general case it accommodates an asymmetric proposal density
# For symetric proposals, the proposal correction factor equals 1, so the logproposal field is set to nothing

immutable MH <: MHSampler
  symmetric::Bool # If symmetric=true then the proposal density is symmetric, else it is asymmetric
  logproposal::Union{Function, Void} # logpdf of asymmetric proposal. For symmetric proposals, logproposal=nothing
  randproposal::Function # random sampling from proposal density

  function MH(symmetric::Bool, logproposal::Union{Function, Void}, randproposal::Function)
    if symmetric && logproposal != nothing
      error("If the symmetric field is true, then logproposal is not used in the calculations")
    end
    new(symmetric, logproposal, randproposal)
  end
end

# Metropolis-Hastings sampler (asymmetric proposal)

MH(logproposal::Function, randproposal::Function) = MH(false, logproposal, randproposal)

# Metropolis sampler (symmetric proposal)

MH(randproposal::Function) = MH(true, nothing, randproposal)

# Random-walk Metropolis, i.e. Metropolis with a normal proposal density

MH{N<:Real}(σ::Matrix{N}) = MH(x::Vector{N} -> rand(MvNormal(x, σ)))
MH{N<:Real}(σ::Vector{N}) = MH(x::Vector{N} -> rand(MvNormal(x, σ)))
MH{N<:Real}(σ::N) = MH(x::N -> rand(Normal(x, σ)))
MH{N<:Real}(::Type{N}=Float64) = MH(x::N -> rand(Normal(x, 1.0)))

### Initialize Metropolis-Hastings sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::MH
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
end

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MH
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
end

## Initialize MHState

sampler_state(sampler::MH, tuner::MCTuner, pstate::ParameterState) =
  MHState(generate_empty(pstate), tuner_state(sampler, tuner))

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::MH
)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  x::Vector{N},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MH
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

## Initialize task

function initialize_task!(
  pstate::ParameterState{Continuous, Univariate},
  sstate::MHState,
  parameter::Parameter{Continuous, Univariate},
  sampler::MH,
  tuner::MCTuner,
  range::BasicMCRange,
  resetplain!::Function,
  iterate!::Function
)
  # Hook inside task to allow remote resetting
  task_local_storage(:reset, resetplain!)

  while true
    iterate!(pstate, sstate, parameter, sampler, tuner, range)
  end
end

function initialize_task!(
  pstate::ParameterState{Continuous, Multivariate},
  sstate::MHState,
  parameter::Parameter{Continuous, Multivariate},
  sampler::MH,
  tuner::MCTuner,
  range::BasicMCRange,
  resetplain!::Function,
  iterate!::Function
)
  # Hook inside task to allow remote resetting
  task_local_storage(:reset, resetplain!)

  while true
    iterate!(pstate, sstate, parameter, sampler, tuner, range)
  end
end

function Base.show(io::IO, sampler::MH)
  issymmetric = sampler.symmetric ? "symmetric" : "non-symmmetric"
  print(io, "MH sampler: $issymmetric proposal")
end

Base.writemime(io::IO, ::MIME"text/plain", sampler::MH) = show(io, sampler)
