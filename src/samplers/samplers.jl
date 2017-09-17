### MCSamplerState represents the Monte Carlo samplers' internal state ("local variables")

abstract type MCSamplerState{F<:VariateForm} end

abstract type MHSamplerState{F<:VariateForm} <: MCSamplerState{F} end

abstract type HMCSamplerState{F<:VariateForm} <: MCSamplerState{F} end

abstract type LMCSamplerState{F<:VariateForm} <: MCSamplerState{F} end

### Abstract Monte Carlo sampler typesystem

## Root Monte Carlo sampler

abstract type MCSampler end

## Family of samplers based on Metropolis-Hastings

abstract type MHSampler <: MCSampler end

## Family of Hamiltonian Monte Carlo samplers

abstract type HMCSampler <: MCSampler end

## Family of Langevin Monte Carlo samplers

abstract type LMCSampler <: MCSampler end

tuner_state(parameter::Parameter, sampler::MCSampler, tuner::VanillaMCTuner) = BasicMCTune(NaN, 0, 0, tuner.period)

tuner_state(parameter::Parameter, sampler::MHSampler, tuner::VanillaMCTuner) = BasicMCTune(1., 0, 0, tuner.period)

tuner_state(parameter::Parameter, sampler::HMCSampler, tuner::VanillaMCTuner) =
  BasicMCTune(sampler.leapstep, 0, 0, tuner.period)

tuner_state(parameter::Parameter, sampler::LMCSampler, tuner::VanillaMCTuner) =
  BasicMCTune(sampler.driftstep, 0, 0, tuner.period)

tuner_state(parameter::Parameter, sampler::MHSampler, tuner::AcceptanceRateMCTuner) = BasicMCTune(1., 0, 0, tuner.period)

tuner_state(parameter::Parameter, sampler::HMCSampler, tuner::AcceptanceRateMCTuner) =
  BasicMCTune(sampler.leapstep, 0, 0, tuner.period)

tuner_state(parameter::Parameter, sampler::LMCSampler, tuner::AcceptanceRateMCTuner) =
  BasicMCTune(sampler.driftstep, 0, 0, tuner.period)

function set_diagnosticindices!(sstate::MCSamplerState, diagnostickeys::Vector{Symbol}, samplerkeys::Vector{Symbol})
  if !isempty(diagnostickeys)
    dindices = map(k -> findfirst(diagnostickeys, k), samplerkeys)

    for (k, i) in zip(samplerkeys, dindices)
      if i != 0
        sstate.diagnosticindices[k] = i
      end
    end
  end
end

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::HMCSampler
)
  pstate.value = x
  parameter.uptogradlogtarget!(pstate)
end

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMCSampler
)
  pstate.value = copy(x)
  parameter.uptogradlogtarget!(pstate)
end

reset!(tune::BasicMCTune, sampler::MCSampler, tuner::VanillaMCTuner) =
  ((tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN))

function reset!(tune::BasicMCTune, sampler::HMCSampler, tuner::AcceptanceRateMCTuner)
  tune.step = sampler.leapstep
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

function reset!(tune::BasicMCTune, sampler::LMCSampler, tuner::AcceptanceRateMCTuner)
  tune.step = sampler.driftstep
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

reset!(
  sstate::MCSamplerState{F},
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MCSampler,
  tuner::MCTuner
) where {S<:ValueSupport, F<:VariateForm} =
  reset!(sstate.tune, sampler, tuner)

hamiltonian(logtarget::Real, momentum::Real) = logtarget-0.5*abs2(momentum)

hamiltonian(logtarget::Real, momentum::RealVector) = logtarget-0.5*dot(momentum, momentum)

function leapfrog!(
  pstate::ParameterState{Continuous, Univariate},
  pstate0::ParameterState{Continuous, Univariate},
  momentum0::Real,
  step::Real,
  gradlogtarget!::Function
)
  local momentum::Real

  momentum = momentum0+0.5*step*pstate0.gradlogtarget
  pstate.value = pstate0.value+step*momentum
  gradlogtarget!(pstate)
  momentum = momentum+0.5*step*pstate.gradlogtarget

  return momentum
end

function leapfrog!(
  pstate::ParameterState{Continuous, Multivariate},
  momentum::RealVector,
  pstate0::ParameterState{Continuous, Multivariate},
  momentum0::RealVector,
  step::Real,
  gradlogtarget!::Function
)
  momentum[:] = momentum0+0.5*step*pstate0.gradlogtarget
  pstate.value[:] = pstate0.value+step*momentum
  gradlogtarget!(pstate)
  momentum[:] = momentum+0.5*step*pstate.gradlogtarget
end

function initialize_step!(
  pstate::ParameterState{Continuous, Univariate},
  pstate0::ParameterState{Continuous, Univariate},
  momentum0::Real,
  step0::Real,
  gradlogtarget!::Function,
  ::Type{DualAveragingMCTuner}
)
  local momentum::Real
  local oldhamiltonian::Real
  local newhamiltonian::Real
  local ratio::Real
  local a::Real
  local step::Real = step0

  oldhamiltonian = hamiltonian(pstate0.logtarget, momentum0)

  momentum = leapfrog!(pstate, pstate0, momentum0, step, gradlogtarget!)
  newhamiltonian = hamiltonian(pstate.logtarget, momentum)

  ratio = newhamiltonian-oldhamiltonian
  a = 2*(exp(ratio) > 0.5)-1

  while exp(ratio)^a > 2^(-a)
    step = (2^a)*step
    momentum = leapfrog!(pstate, pstate, momentum, step, gradlogtarget!)
    newhamiltonian = hamiltonian(pstate.logtarget, momentum)

    ratio = newhamiltonian-oldhamiltonian
  end

  return step
end

function initialize_step!(
  pstate::ParameterState{Continuous, Multivariate},
  momentum::RealVector,
  pstate0::ParameterState{Continuous, Multivariate},
  momentum0::RealVector,
  step0::Real,
  gradlogtarget!::Function,
  ::Type{DualAveragingMCTuner}
)
  local oldhamiltonian::Real
  local newhamiltonian::Real
  local ratio::Real
  local a::Real
  local step::Real = step0

  oldhamiltonian = hamiltonian(pstate0.logtarget, momentum0)

  leapfrog!(pstate, momentum, pstate0, momentum0, step, gradlogtarget!)
  newhamiltonian = hamiltonian(pstate.logtarget, momentum)

  ratio = newhamiltonian-oldhamiltonian
  a = 2*(exp(ratio) > 0.5)-1

  while exp(ratio)^a > 2^(-a)
    step = (2^a)*step
    leapfrog!(pstate, momentum, pstate, moment, step, gradlogtarget!)
    newhamiltonian = hamiltonian(pstate.logtarget, momentum)

    ratio = newhamiltonian-oldhamiltonian
  end

  return step
end
