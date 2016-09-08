### MCSamplerState represents the Monte Carlo samplers' internal state ("local variables")

abstract MCSamplerState{F<:VariateForm}

abstract MHSamplerState{F<:VariateForm} <: MCSamplerState{F}

abstract HMCSamplerState{F<:VariateForm} <: MCSamplerState{F}

abstract LMCSamplerState{F<:VariateForm} <: MCSamplerState{F}

### Abstract Monte Carlo sampler typesystem

## Root Monte Carlo sampler

abstract MCSampler

## Family of samplers based on Metropolis-Hastings

abstract MHSampler <: MCSampler

## Family of Hamiltonian Monte Carlo samplers

abstract HMCSampler <: MCSampler

## Family of Langevin Monte Carlo samplers

abstract LMCSampler <: MCSampler

### Code generation of sampler fields

function codegen_setfield(sampler::MCSampler, field::Symbol, f::Function)
  @gensym setfield
  quote
    function $setfield(_state, _states::VariableStateVector)
      setfield!($sampler, $(QuoteNode(field)), $(f)(_state, _states))
    end
  end
end

function initialize_step!{F<:VariateForm}(
  sstate::HMCSamplerState{F},
  parameter::Parameter{Continuous, F},
  sampler::HMCSampler,
  tuner::DualAveragingMCTuner
)
  sstate.oldhamiltonian = hamiltonian(sstate.pstate.logtarget, sstate.momentum)

  leapfrog!(sstate, parameter)
  sstate.newhamiltonian = hamiltonian(sstate.pstate.logtarget, sstate.momentum)

  sstate.ratio = sstate.newhamiltonian-sstate.oldhamiltonian
  sstate.a = 2*(exp(sstate.ratio) > 0.5)-1

  while exp(sstate.ratio)^sstate.a > 2^(-sstate.a)
    sstate.tune.step = (2^sstate.a)*sstate.tune.step
    leapfrog!(sstate, parameter)
    sstate.newhamiltonian = hamiltonian(sstate.pstate.logtarget, sstate.momentum)

    sstate.ratio = sstate.newhamiltonian-sstate.oldhamiltonian
  end
end

function initialize_step!(
  sstate::HMCSamplerState{Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::HMCSampler,
  tuner::DualAveragingMCTuner,
  momentum::Real
)
  sstate.momentum = momentum # randn()
  initialize_step!(sstate, parameter, sampler, tuner)
end

function initialize_step!(
  sstate::HMCSamplerState{Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMCSampler,
  tuner::DualAveragingMCTuner,
  momentum::RealVector
)
  sstate.momentum = momentum
  initialize_step!(sstate, parameter, sampler, tuner)
end

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

reset!{S<:ValueSupport, F<:VariateForm}(
  sstate::MCSamplerState{F},
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MCSampler,
  tuner::MCTuner
) =
  reset!(sstate.tune, sampler, tuner)
