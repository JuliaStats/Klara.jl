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

function initialize!{S<:ValueSupport, F<:VariateForm}(pstate::ParameterState{S, F}, parameter::Parameter{S, F}, sampler::MH)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of parameter support"
end

## Initialize MHState

sampler_state(sampler::MH, tuner::MCTuner, pstate::ParameterState) =
  MHState(generate_empty(pstate), tuner_state(sampler, tuner))

## Reset parameter state

function reset!{S<:ValueSupport}(pstate::ParameterState{S, Univariate}, x, parameter::Parameter{S, Univariate}, sampler::MH)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!{S<:ValueSupport}(
  pstate::ParameterState{S, Multivariate},
  x,
  parameter::Parameter{S, Multivariate},
  sampler::MH
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

function Base.show(io::IO, sampler::MH)
  issymmetric = sampler.symmetric ? "symmetric" : "non-symmmetric"
  print(io, "MH sampler: $issymmetric proposal")
end

Base.writemime(io::IO, ::MIME"text/plain", sampler::MH) = show(io, sampler)
