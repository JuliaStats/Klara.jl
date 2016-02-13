### Abstract RAM state

abstract RAMState <: MCSamplerState

### RAM state subtypes

## UnvRAMState holds the internal state ("local variables") of the RAM sampler for univariate parameters

type UnvRAMState <: RAMState
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by RAM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  S::Real
  SST::Real
  randnsample::Real
  η::Real

  function UnvRAMState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    S::Real,
    SST::Real,
    randnsample::Real,
    η::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, S, SST, randnsample, η)
  end
end

UnvRAMState(
  pstate::ParameterState{Continuous, Univariate},
  tune::MCTunerState=VanillaMCTune(),
  S::Real=NaN,
  SST::Real=abs2(S)
) =
  UnvRAMState(pstate, tune, NaN, S, SST, NaN, NaN)

## MuvRAMState holds the internal state ("local variables") of the RAM sampler for multivariate parameters

type MuvRAMState{N<:Real} <: RAMState
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by RAM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  S::Matrix{N}
  SST::Matrix{N}
  randnsample::Vector{N}
  η::Real

  function MuvRAMState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    S::Matrix{N},
    SST::Matrix{N},
    randnsample::Vector{N},
    η::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, S, SST, randnsample, η)
  end
end

MuvRAMState{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  tune::MCTunerState,
  ratio::Real,
  S::Matrix{N},
  SST::Matrix{N},
  randnsample::Vector{N},
  η::Real
) =
  MuvRAMState{N}(pstate, tune, ratio, S, SST, randnsample, η)

MuvRAMState{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  tune::MCTunerState=VanillaMCTune(),
  S::Matrix{N}=Array(eltype(pstate), pstate.size, pstate.size),
  SST::Matrix{N}=S*S'
) =
  MuvRAMState(pstate, tune, NaN, S, SST, Array(eltype(pstate), pstate.size), NaN)

Base.eltype{N<:Real}(::Type{MuvRAMState{N}}) = N
Base.eltype{N<:Real}(s::MuvRAMState{N}) = N

### Robust adaptive Metropolis (RAM) sampler

immutable RAM{N<:Real} <: MHSampler
  S0::Matrix{N} # Initial adaptation matrix
  targetrate::Real # Target acceptance rate
  γ::Real # Exponent for scaling stepsize η

  function RAM(S0::Matrix{N}, targetrate::Real, γ::Real)
    @assert all(i -> i > 0, diag(S0)) "All diagonal elements of initial adaptation matrix must be positive"
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert 0.5 < γ <= 1 "Exponent of stepsize must be greater than 0.5 and less or equal to 1"
    new(S0, targetrate, γ)
  end
end

RAM{N<:Real}(S0::Matrix{N}, targetrate::Real=0.234, γ::Real=0.7) = RAM{N}(S0, targetrate, γ)

RAM{N<:Real}(S0::Vector{N}, targetrate::Real=0.234, γ::Real=0.7) = RAM(diagm(S0), targetrate, γ)

RAM(S0::Real=1., n::Int=1, targetrate::Real=0.234, γ::Real=0.7) = RAM(fill(S0, n), targetrate, γ)

### Initialize RAM sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::RAM
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of parameter support"
end

## Initialize MuvRAMState

sampler_state(sampler::RAM, tuner::MCTuner, pstate::ParameterState{Continuous, Univariate}) =
  UnvRAMState(
    generate_empty(pstate),
    tuner_state(sampler, tuner),
    sampler.S0[1, 1],
    NaN
  )

sampler_state(sampler::RAM, tuner::MCTuner, pstate::ParameterState{Continuous, Multivariate}) =
  MuvRAMState(
    generate_empty(pstate),
    tuner_state(sampler, tuner),
    sampler.S0,
    Array(eltype(pstate), pstate.size, pstate.size)
  )

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::RAM
)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  x::Vector{N},
  parameter::Parameter{Continuous, Multivariate},
  sampler::RAM
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

Base.show(io::IO, sampler::RAM) = print(io, "RAM sampler: target rate = $(sampler.targetrate), γ = $(sampler.γ)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::RAM) = show(io, sampler)
