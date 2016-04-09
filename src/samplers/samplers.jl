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

function codegen_proposal_method_via_distribution(sampler::MCSampler, distribution::Symbol, f::Function)
  @gensym proposal_method_via_distribution
  quote
    function $proposal_method_via_distribution(_state, _states::VariableStateVector)
      $(f)(getfield($sampler, $(QuoteNode(distribution))), _state.value)
    end
  end
end

tuner_state(sampler::MCSampler, tuner::VanillaMCTuner) = VanillaMCTune(0, 0, tuner.period)

tuner_state(sampler::MHSampler, tuner::AcceptanceRateMCTuner) = AcceptanceRateMCTune(1., 0, 0, tuner.period)
tuner_state(sampler::HMCSampler, tuner::AcceptanceRateMCTuner) = AcceptanceRateMCTune(sampler.leapstep, 0, 0, tuner.period)
tuner_state(sampler::LMCSampler, tuner::AcceptanceRateMCTuner) = AcceptanceRateMCTune(sampler.driftstep, 0, 0, tuner.period)

reset!(tune::VanillaMCTune, sampler::MCSampler, tuner::VanillaMCTuner) =
  ((tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN))

function reset!(tune::AcceptanceRateMCTune, sampler::HMCSampler, tuner::AcceptanceRateMCTuner)
  tune.step = sampler.leapstep
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

function reset!(tune::AcceptanceRateMCTune, sampler::LMCSampler, tuner::AcceptanceRateMCTuner)
  tune.step = sampler.driftstep
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
