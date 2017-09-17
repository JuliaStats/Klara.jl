### Reference:
### Matti Vihola
### Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate
### Statistics and Computing, 2012, 22 (5), pp 997-1008

### Abstract RAM state

abstract type RAMState{F<:VariateForm} <: MHSamplerState{F} end

### RAM state subtypes

## UnvRAMState holds the internal state ("local variables") of the RAM sampler for univariate parameters

mutable struct UnvRAMState <: RAMState{Univariate}
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by RAM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  S::Real
  SST::Real
  randnsample::Real
  η::Real
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function UnvRAMState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    S::Real,
    SST::Real,
    randnsample::Real,
    η::Real,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(pstate, tune, ratio, S, SST, randnsample, η, count, diagnosticindices)
  end
end

UnvRAMState(
  pstate::ParameterState{Continuous, Univariate},
  tune::MCTunerState=BasicMCTune(),
  S::Real=NaN,
  SST::Real=abs2(S)
) =
  UnvRAMState(pstate, tune, NaN, S, SST, NaN, NaN, 0, Dict{Symbol, Integer}())

## MuvRAMState holds the internal state ("local variables") of the RAM sampler for multivariate parameters

mutable struct MuvRAMState <: RAMState{Multivariate}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by RAM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  S::RealLowerTriangular
  SST::RealMatrix
  randnsample::RealVector
  η::Real
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function MuvRAMState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    S::RealLowerTriangular,
    SST::RealMatrix,
    randnsample::RealVector,
    η::Real,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(pstate, tune, ratio, S, SST, randnsample, η, count, diagnosticindices)
  end
end

MuvRAMState(
  pstate::ParameterState{Continuous, Multivariate},
  tune::MCTunerState=BasicMCTune(),
  S::RealLowerTriangular=RealLowerTriangular(Array{eltype(pstate)}(pstate.size, pstate.size)),
  SST::RealMatrix=S*S'
) =
  MuvRAMState(pstate, tune, NaN, S, SST, Array{eltype(pstate)}(pstate.size), NaN, 0, Dict{Symbol, Integer}())

### Robust adaptive Metropolis (RAM) sampler

struct RAM <: MHSampler
  S0::RealLowerTriangular # Initial adaptation matrix
  targetrate::Real # Target acceptance rate
  γ::Real # Exponent for scaling stepsize η

  function RAM(S0::RealLowerTriangular, targetrate::Real, γ::Real)
    @assert all(i -> i > 0, diag(S0)) "All diagonal elements of initial adaptation matrix must be positive"
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert 0.5 < γ <= 1 "Exponent of stepsize must be greater than 0.5 and less or equal to 1"
    new(S0, targetrate, γ)
  end
end

RAM(S0::RealMatrix; targetrate::Real=0.234, γ::Real=0.7) = RAM(RealLowerTriangular(S0), targetrate, γ)

RAM(S0::RealVector; targetrate::Real=0.234, γ::Real=0.7) = RAM(RealLowerTriangular(diagm(S0)), targetrate, γ)

RAM(S0::Real=1., n::Integer=1; targetrate::Real=0.234, γ::Real=0.7) = RAM(fill(S0, n), targetrate=targetrate, γ=γ)

### Initialize RAM sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::RAM,
  outopts::Dict
) where F<:VariateForm
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

## Initialize RAM state

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::RAM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvRAMState( generate_empty(pstate), tuner_state(parameter, sampler, tuner), sampler.S0[1, 1], NaN)
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::RAM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvRAMState(
    generate_empty(pstate),
    tuner_state(parameter, sampler, tuner),
    copy(sampler.S0),
    Array{eltype(pstate)}(pstate.size, pstate.size)
  )
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

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

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::RAM
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

## Reset sampler state

function reset!(
  sstate::RAMState{Univariate},
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::RAM,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.S = sampler.S0[1, 1]
  sstate.count = 0
end

function reset!(
  sstate::RAMState{Multivariate},
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::RAM,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.S = copy(sampler.S0)
  sstate.count = 0
end

show(io::IO, sampler::RAM) = print(io, "RAM sampler: target rate = $(sampler.targetrate), γ = $(sampler.γ)")
