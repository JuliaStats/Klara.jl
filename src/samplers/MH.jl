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

# In its most general case it accommodates an asymmetric proposal distribution
# For symmetric proposals, the proposal correction factor equals 1, so the logproposal field is set to nothing
# For non-normalised proposals, the lognormalise() method is used for calculating the proposal correction factor

type MH <: MHSampler
  symmetric::Bool # If symmetric=true then the proposal distribution is symmetric, else it is asymmetric
  normalised::Bool # If normalised=true then the proposal distribution is normalised, else it is non-normalised
  proposal::Union{Distribution, Void} # Proposal distribution
  setproposal::Function # Function for setting the proposal distribution
  logproposal::Union{Distribution, Void} # logpdf of asymmetric proposal. For symmetric proposals, logproposal=nothing

  function MH(symmetric::Bool, normalised::Bool, setproposal::Function)
    instance = new()
    
    instance.symmetric = symmetric
    instance.normalised = normalised
    instance.setproposal = eval(codegen_setfield(instance, :proposal, setproposal))   
    instance.logproposal = symmetric ? nothing : eval(codegen_proposal_method_via_distribution(instance, :proposal, logpdf))
            
    instance
  end
end

MH(setproposal::Function; signature::Symbol=:high, args...) = MH(setproposal, Val{signature}; args...)

MH(setproposal::Function, ::Type{Val{:low}}; symmetric::Bool=true, normalised::Bool=true) =
  MH(symmetric, normalised, setproposal)

MH(setproposal::Function, ::Type{Val{:high}}; symmetric::Bool=true, normalised::Bool=true, nkeys::Int=0, vfarg::Bool=false) =
  MH(symmetric, normalised, eval(codegen_lowlevel_variable_method(setproposal, Symbol[], nothing, nkeys, vfarg)))

# Random-walk Metropolis, i.e. Metropolis with a normal proposal distribution

MH{N<:Real}(σ::Matrix{N}) = MH(x::Vector{N} -> MvNormal(x, σ), signature=:high)
MH{N<:Real}(σ::Vector{N}) = MH(x::Vector{N} -> MvNormal(x, σ), signature=:high)
MH{N<:Real}(σ::N) = MH(x::N -> Normal(x, σ), signature=:high)
MH{N<:Real}(::Type{N}=Float64) = MH(x::N -> Normal(x, 1.0), signature=:high)

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
  isnormalised = sampler.normalised ? "normalised" : "non-normalised"
  print(io, "MH sampler: $issymmetric $isnormalised proposal")
end

Base.writemime(io::IO, ::MIME"text/plain", sampler::MH) = show(io, sampler)
