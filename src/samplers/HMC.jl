### Abstract HMC state

abstract HMCState{F<:VariateForm} <: HMCSamplerState{F}

### HMC state subtypes

## UnvHMCState holds the internal state ("local variables") of the HMC sampler for univariate parameters

type UnvHMCState <: HMCState{Univariate}
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by HMC
  tune::MCTunerState
  nleaps::Integer
  ratio::Real
  a::Real
  momentum::Real
  oldhamiltonian::Real
  newhamiltonian::Real
  count::Integer

  function UnvHMCState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    nleaps::Integer,
    ratio::Real,
    a::Real,
    momentum::Real,
    oldhamiltonian::Real,
    newhamiltonian::Real,
    count::Integer
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(a)
      @assert 0 <= a <= 1 "Acceptance probability should be in [0, 1]"
    end
    new(pstate, tune, nleaps, ratio, a, momentum, oldhamiltonian, newhamiltonian, count)
  end
end

UnvHMCState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune(), nleaps::Integer=0) =
  UnvHMCState(pstate, tune, nleaps, NaN, NaN, NaN, NaN, NaN, 0)

## MuvHMCState holds the internal state ("local variables") of the HMC sampler for multivariate parameters

type MuvHMCState <: HMCState{Multivariate}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by HMC
  tune::MCTunerState
  nleaps::Integer
  ratio::Real
  a::Real
  momentum::RealVector
  oldhamiltonian::Real
  newhamiltonian::Real
  count::Integer

  function MuvHMCState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    nleaps::Integer,
    ratio::Real,
    a::Real,
    momentum::RealVector,
    oldhamiltonian::Real,
    newhamiltonian::Real,
    count::Integer
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(a)
      @assert 0 <= a <= 1 "Acceptance probability should be in [0, 1]"
    end
    new(pstate, tune, nleaps, ratio, a, momentum, oldhamiltonian, newhamiltonian, count)
  end
end

MuvHMCState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune(), nleaps::Integer=0) =
  MuvHMCState(pstate, tune, nleaps, NaN, NaN, Array(eltype(pstate), pstate.size), NaN, NaN, 0)

### Hamiltonian Monte Carlo (HMC)

immutable HMC <: HMCSampler
  leapstep::Real
  nleaps::Integer

  function HMC(leapstep::Real, nleaps::Integer)
    @assert leapstep > 0 "Leapfrog step is not positive"
    @assert nleaps > 0 "Number of leapfrog steps is not positive"
    new(leapstep, nleaps)
  end
end

HMC(leapstep::Real=0.1) = HMC(leapstep, 10)

### Initialize HMC sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::HMC
)
  parameter.uptogradlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
end

## Initialize HMC state

tuner_state(parameter::Parameter, sampler::HMC, tuner::DualAveragingMCTuner) =
  DualAveragingMCTune(
  step=sampler.leapstep,
  λ=sampler.nleaps*sampler.leapstep,
  εbar=tuner.ε0bar,
  hbar=tuner.h0bar,
  accepted=0,
  proposed=0,
  totproposed=tuner.period
)

sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::HMC,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
) =
  UnvHMCState(generate_empty(pstate), tuner_state(parameter, sampler, tuner), sampler.nleaps)

sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMC,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
) =
  MuvHMCState(generate_empty(pstate), tuner_state(parameter, sampler, tuner), sampler.nleaps)

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
)
  sstate = MuvHMCState(generate_empty(pstate), tuner_state(parameter, sampler, tuner), 0)
  initialize_step!(sstate, parameter, sampler, tuner, randn())
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvHMCState(generate_empty(pstate), tuner_state(parameter, sampler, tuner), 0)
  initialize_step!(sstate, parameter, sampler, tuner, randn(sstate.pstate.size))
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate
end

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::HMC
)
  pstate.value = x
  parameter.uptogradlogtarget!(pstate)
end

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMC
)
  pstate.value = copy(x)
  parameter.uptogradlogtarget!(pstate)
end

function reset!(tune::DualAveragingMCTune, sampler::HMC, tuner::DualAveragingMCTuner)
  tune.step = 1
  tune.λ = sampler.nleaps*sampler.leapstep
  tune.εbar = tuner.ε0bar
  tune.hbar = tuner.h0bar
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

function reset!(
  sstate::MuvHMCState,
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner
)
  reset!(sstate.tune, sampler, tuner)
  initialize_step!(sstate, parameter, sampler, tuner, randn())
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate.count = 0
end

function reset!(
  sstate::MuvHMCState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner
)
  reset!(sstate.tune, sampler, tuner)
  initialize_step!(sstate, parameter, sampler, tuner, randn(sstate.pstate.size))
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate.count = 0
end

### Compute Hamiltonian

hamiltonian(logtarget::Real, momentum::Real) = logtarget-0.5*abs2(momentum)

hamiltonian(logtarget::Real, momentum::RealVector) = logtarget-0.5*dot(momentum, momentum)

### Perform leapfrog iteration

function leapfrog!{F<:VariateForm}(sstate::HMCState, parameter::Parameter{Continuous, F})
  sstate.momentum += 0.5*sstate.tune.step*sstate.pstate.gradlogtarget
  sstate.pstate.value += sstate.tune.step*sstate.momentum
  parameter.gradlogtarget!(sstate.pstate)
  sstate.momentum += 0.5*sstate.tune.step*sstate.pstate.gradlogtarget
end

Base.show(io::IO, sampler::HMC) =
  print(io, "HMC sampler: number of leaps = $(sampler.nleaps), leap step = $(sampler.leapstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::HMC) = show(io, sampler)
