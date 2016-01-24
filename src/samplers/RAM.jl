### Abstract RAM state

abstract RAMState <: MCSamplerState

### RAM state subtypes

## MuvRAMState holds the internal state ("local variables") of the RAM sampler for multivariate parameters

type MuvRAMState{N<:Real} <: RAMState
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by RAM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  S::Matrix{N}
  SSΤ::Matrix{N}
  randnsample::Vector{N}
  η::Real

  function MuvRAMState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    S::Matrix{N},
    SSΤ::Matrix{N},
    randnsample::Vector{N},
    η::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, S, SSΤ, randnsample, η)
  end
end

MuvRAMState(
  pstate::ParameterState{Continuous, Multivariate},
  tune::MCTunerState=VanillaMCTune(),
  S::Matrix{N}=Array(eltype(pstate), pstate.size, pstate.size),
  SST::Matrix{N}=S*S'
) =
  MuvRAMState(pstate, tune, NaN, S, SST, Array(eltype(pstate), pstate.size), NaN)

Base.eltype{N<:Real}(::Type{MuvRAMState{N}}) = N
Base.eltype{N<:Real}(s::v{N}) = N

### Robust adaptive Metropolis (RAM) sampler

immutable RAM{N<:Real} <: MHSampler
  S0::Matrix{N} # Initial adaptation matrix
  targetrate::Real # Target acceptance rate

  function RAM(S0::Matrix{N}, targetrate::Real)
    @assert all(i -> i > 0, diag(S0)) "All diagonal elements of initial adaptation matrix must be positive"
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    new(S0, targetrate)
  end
end

RAM{N<:Real}(S0::Matrix{N}, targetrate::Real) = RAM{N}(S0, targetrate)

RAM{N<:Real}(S0::Matrix{N}) = RAM(S0, 0.234)

RAM{N<:Real}(S0::Vector{N}, targetrate::Real=0.234) = RAM(diagm(S0), targetrate)

# RAM(S0::Real, n::Int=1, targetrate::Real=0.234) = RAM(fill(s0, n), 0.234)

### Initialize RAM sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::RAM
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
end

## Initialize MuvRAMState

sampler_state(sampler::RAM, tuner::MCTuner, pstate::ParameterState{Continuous, Multivariate}) =
  MuvRAMState(generate_empty(pstate), sampler.driftstep, tuner_state(sampler, tuner))
  MuvRAMState(
    generate_empty(pstate),
    tuner_state(sampler, tuner),
    sampler.S0,
    Array(eltype(pstate), pstate.size, pstate.size)
  )

## Reset parameter state

function reset!{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  x::Vector{N},
  parameter::Parameter{Continuous, Multivariate},
  sampler::RAM
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

Base.show(io::IO, sampler::RAM) = print(io, "RAM sampler")

Base.writemime(io::IO, ::MIME"text/plain", sampler::RAM) = show(io, sampler)
