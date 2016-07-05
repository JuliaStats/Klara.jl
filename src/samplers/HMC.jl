### Abstract HMC state

abstract HMCState <: MCSamplerState

### HMC state subtypes

## UnvHMCState holds the internal state ("local variables") of the HMC sampler for univariate parameters

type UnvHMCState <: HMCState
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by HMC
  tune::MCTunerState
  ratio::Real
  momentum::Real
  oldhamiltonian::Real
  newhamiltonian::Real

  function UnvHMCState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    momentum::Real,
    oldhamiltonian::Real,
    newhamiltonian::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, momentum, oldhamiltonian, newhamiltonian)
  end
end

UnvHMCState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune()) =
  UnvHMCState(pstate, tune, NaN, NaN, NaN, NaN)

## MuvHMCState holds the internal state ("local variables") of the HMC sampler for multivariate parameters

type MuvHMCState <: HMCState
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by HMC
  tune::MCTunerState
  ratio::Real
  momentum::RealVector
  oldhamiltonian::Real
  newhamiltonian::Real

  function MuvHMCState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    momentum::RealVector,
    oldhamiltonian::Real,
    newhamiltonian::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, tune, ratio, momentum, oldhamiltonian, newhamiltonian)
  end
end

MuvHMCState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvHMCState(pstate, tune, NaN, Array(eltype(pstate), pstate.size), NaN, NaN)

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

sampler_state(sampler::HMC, tuner::MCTuner, pstate::ParameterState{Continuous, Univariate}, vstate::VariableStateVector) =
  UnvHMCState(generate_empty(pstate), tuner_state(sampler, tuner))

sampler_state(sampler::HMC, tuner::MCTuner, pstate::ParameterState{Continuous, Multivariate}, vstate::VariableStateVector) =
  MuvHMCState(generate_empty(pstate), tuner_state(sampler, tuner))

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

### Compute Hamiltonian

hamiltonian(logtarget::Real, momentum::Real) = -logtarget+0.5*abs2(momentum)

hamiltonian(logtarget::Real, momentum::RealVector) = -logtarget+0.5*dot(momentum, momentum)

### Perform leapfrog iteration

function leapfrog!{F<:VariateForm}(sstate::HMCState, parameter::Parameter{Continuous, F}, stepsize::Real)
  sstate.momentum += 0.5*stepsize*sstate.pstate.gradlogtarget
  sstate.pstate.value += stepsize*sstate.momentum
  parameter.gradlogtarget!(sstate.pstate)
  sstate.momentum += 0.5*stepsize*sstate.pstate.gradlogtarget
end

Base.show(io::IO, sampler::HMC) =
  print(io, "HMC sampler: number of leaps = $(sampler.nleaps), leap step = $(sampler.leapstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::HMC) = show(io, sampler)
