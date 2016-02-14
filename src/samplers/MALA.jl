### Abstract MALA state

abstract MALAState <: MCSamplerState

### MALA state subtypes

## UnvMALAState holds the internal state ("local variables") of the MALA sampler for univariate parameters

type UnvMALAState <: MALAState
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by MALA
  tune::MCTunerState
  ratio::Real
  vmean::Real
  pnewgivenold::Real
  poldgivennew::Real

  function UnvMALAState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    vmean::Real,
    pnewgivenold::Real,
    poldgivennew::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, vmean, pnewgivenold, poldgivennew)
  end
end

UnvMALAState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=VanillaMCTune()) =
  UnvMALAState(pstate, tune, NaN, NaN, NaN, NaN)

## MuvMALAState holds the internal state ("local variables") of the MALA sampler for multivariate parameters

type MuvMALAState{N<:Real} <: MALAState
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by MALA
  tune::MCTunerState
  ratio::Real
  vmean::Vector{N}
  pnewgivenold::Real
  poldgivennew::Real

  function MuvMALAState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    vmean::Vector{N},
    pnewgivenold::Real,
    poldgivennew::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, vmean, pnewgivenold, poldgivennew)
  end
end

MuvMALAState{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  tune::MCTunerState,
  ratio::Real,
  vmean::Vector{N},
  pnewgivenold::Real,
  poldgivennew::Real
) =
  MuvMALAState{N}(pstate, tune, ratio, vmean, pnewgivenold, poldgivennew)

MuvMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=VanillaMCTune()) =
  MuvMALAState(pstate, tune, NaN, Array(eltype(pstate), pstate.size), NaN, NaN)

Base.eltype{N<:Real}(::Type{MuvMALAState{N}}) = N
Base.eltype{N<:Real}(s::MuvMALAState{N}) = N

### Metropolis-adjusted Langevin Algorithm (MALA)

immutable MALA <: LMCSampler
  driftstep::Real

  function MALA(driftstep::Real)
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep)
  end
end

MALA() = MALA(1.)

### Initialize MALA sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::MALA
)
  parameter.uptogradlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of parameter support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of parameter support"
end

## Initialize MALA state

sampler_state(sampler::MALA, tuner::MCTuner, pstate::ParameterState{Continuous, Univariate}) =
  UnvMALAState(generate_empty(pstate), tuner_state(sampler, tuner))

sampler_state(sampler::MALA, tuner::MCTuner, pstate::ParameterState{Continuous, Multivariate}) =
  MuvMALAState(generate_empty(pstate), tuner_state(sampler, tuner))

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA
)
  pstate.value = x
  parameter.uptogradlogtarget!(pstate)
end

function reset!{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  x::Vector{N},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA
)
  pstate.value = copy(x)
  parameter.uptogradlogtarget!(pstate)
end

Base.show(io::IO, sampler::MALA) = print(io, "MALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::MALA) = show(io, sampler)
