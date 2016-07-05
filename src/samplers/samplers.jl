### MCSamplerState represents the Monte Carlo samplers' internal state ("local variables")

abstract MCSamplerState

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

tuner_state(sampler::MHSampler, tuner::VanillaMCTuner) = BasicMCTune(1., 0, 0, tuner.period)
tuner_state(sampler::HMCSampler, tuner::VanillaMCTuner) = BasicMCTune(sampler.leapstep, 0, 0, tuner.period)
tuner_state(sampler::LMCSampler, tuner::VanillaMCTuner) = BasicMCTune(sampler.driftstep, 0, 0, tuner.period)

tuner_state(sampler::MHSampler, tuner::AcceptanceRateMCTuner) = BasicMCTune(1., 0, 0, tuner.period)
tuner_state(sampler::HMCSampler, tuner::AcceptanceRateMCTuner) = BasicMCTune(sampler.leapstep, 0, 0, tuner.period)
tuner_state(sampler::LMCSampler, tuner::AcceptanceRateMCTuner) = BasicMCTune(sampler.driftstep, 0, 0, tuner.period)

tuner_state(sampler::MHSampler, tuner::RobertsRosenthalMCTuner) = RobertsRosenthalMCTune(1., true, 0, 0, tuner.period)

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

function reset!(tune::RobertsRosenthalMCTune, sampler::MHSampler, tuner::RobertsRosenthalMCTuner)
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

reset!{S<:ValueSupport, F<:VariateForm}(
  sstate::MCSamplerState,
  pstate::ParameterState{S, F},
  parameter::Parameter{S, F},
  sampler::MCSampler,
  tuner::MCTuner
) =
  reset!(sstate.tune, sampler, tuner)
