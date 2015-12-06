### Abstract MALA state

abstract MALAState <: MCSamplerState

### MALA state subtypes

## UnvMALAState holds the internal state ("local variables") of the MALA sampler for univariate parameters

type UnvMALAState <: MALAState
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by MALA
  driftstep::Real # Drift stepsize for a single Monte Carlo iteration
  tune::MCTunerState
  ratio::Real
  vmean::Real
  pnewgivenold::Real
  poldgivennew::Real

  function UnvMALAState(
    pstate::ParameterState{Continuous, Univariate},
    driftstep::Real,
    tune::MCTunerState,
    ratio::Real,
    vmean::Real,
    pnewgivenold::Real,
    poldgivennew::Real
  )
    if !isnan(driftstep)
      @assert driftstep > 0 "Drift step size must be positive"
    end
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, driftstep, tune, ratio, vmean, pnewgivenold, poldgivennew)
  end
end

UnvMALAState(pstate::ParameterState{Continuous, Univariate}, driftstep::Real=1., tune::MCTunerState=VanillaMCTune()) =
  UnvMALAState(pstate, driftstep, tune, NaN, NaN, NaN, NaN)

## MuvMALAState holds the internal state ("local variables") of the MALA sampler for multivariate parameters

type MuvMALAState{N<:Real} <: MALAState
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by MALA
  driftstep::Real # Drift stepsize for a single Monte Carlo iteration
  tune::MCTunerState
  ratio::Real
  vmean::Vector{N}
  pnewgivenold::Real
  poldgivennew::Real

  function MuvMALAState(
    pstate::ParameterState{Continuous, Multivariate},
    driftstep::Real,
    tune::MCTunerState,
    ratio::Real,
    vmean::Vector{N},
    pnewgivenold::Real,
    poldgivennew::Real
  )
    if !isnan(driftstep)
      @assert driftstep > 0 "Drift step size must be positive"
    end
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, driftstep, tune, ratio, vmean, pnewgivenold, poldgivennew)
  end
end

MuvMALAState{N<:Real}(
  pstate::ParameterState{Continuous, Multivariate},
  driftstep::Real,
  tune::MCTunerState,
  ratio::Real,
  vmean::Vector{N},
  pnewgivenold::Real,
  poldgivennew::Real
) =
  MuvMALAState{N}(pstate, driftstep, tune, ratio, vmean, pnewgivenold, poldgivennew)

MuvMALAState(pstate::ParameterState{Continuous, Multivariate}, driftstep::Real=1., tune::MCTunerState=VanillaMCTune()) =
  MuvMALAState(pstate, driftstep, tune, NaN, Array(eltype(pstate), pstate.size), NaN, NaN)

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

function initialize!{S<:VariableState}(
  pstate::ParameterState{Continuous, Univariate},
  vstate::Vector{S},
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA
)
  parameter.uptogradlogtarget!(pstate, vstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
  @assert isfinite(pstate.gradlogtarget) "Gradient of log-target not finite: initial values out of parameter support"
end

function initialize!{S<:VariableState}(
  pstate::ParameterState{Continuous, Multivariate},
  vstate::Vector{S},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA
)
  parameter.uptogradlogtarget!(pstate, vstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial values out of parameter support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of parameter support"
end

## Initialize MuvMALAState

sampler_state(sampler::MALA, tuner::MCTuner, pstate::ParameterState{Continuous, Univariate}) =
  UnvMALAState(generate_empty(pstate), sampler.driftstep, tuner_state(sampler, tuner))

sampler_state(sampler::MALA, tuner::MCTuner, pstate::ParameterState{Continuous, Multivariate}) =
  MuvMALAState(generate_empty(pstate), sampler.driftstep, tuner_state(sampler, tuner))

## Reset parameter state

function reset!{S<:VariableState}(
  pstate::ParameterState{Continuous, Univariate},
  vstate::Vector{S},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA
)
  pstate.value = x
  parameter.uptogradlogtarget!(pstate, vstate)
end

function reset!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Multivariate},
  vstate::Vector{S},
  x::Vector{N},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA
)
  pstate.value = copy(x)
  parameter.uptogradlogtarget!(pstate, vstate)
end

## Initialize task

function initialize_task!{S<:VariableState}(
  pstate::ParameterState{Continuous, Univariate},
  vstate::Vector{S},
  sstate::UnvMALAState,
  parameter::Parameter{Continuous, Univariate},
  sampler::MALA,
  tuner::MCTuner,
  range::BasicMCRange,
  resetplain!::Function,
  iterate!::Function
)
  # Hook inside task to allow remote resetting
  task_local_storage(:reset, resetplain!)

  while true
    iterate!(pstate, vstate, sstate, parameter, sampler, tuner, range)
  end
end

function initialize_task!{N<:Real, S<:VariableState}(
  pstate::ParameterState{Continuous, Multivariate},
  vstate::Vector{S},
  sstate::MuvMALAState{N},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MALA,
  tuner::MCTuner,
  range::BasicMCRange,
  resetplain!::Function,
  iterate!::Function
)
  # Hook inside task to allow remote resetting
  task_local_storage(:reset, resetplain!)

  while true
    iterate!(pstate, vstate, sstate, parameter, sampler, tuner, range)
  end
end
