### Plain Monte Carlo jobs do not use tasks

type PlainMCJob <: MCJob
  send::Function
  receive::Function
end

PlainMCJob() = PlainMCJob(identity, ()->())

PlainMCJob(model::MCModel, sampler::HMCSampler, runner::SerialMC, tuner::MCTuner) =
  PlainMCJob(identity, ()->iterate!(initialize(model, sampler, runner, tuner), model, sampler, runner, tuner, identity))
