### Plain Monte Carlo jobs do not use tasks

type PlainMCJob <: MCJob
  send::Function
  receive::Function
end

PlainMCJob() = PlainMCJob(identity, ()->())

function PlainMCJob(model::MCModel, sampler::HMCSampler, runner::SerialMC, tuner::MCTuner)
  stash::HMCStash = initialize(model, sampler, runner, tuner)
  PlainMCJob(identity, ()->iterate!(stash, model, sampler, runner, tuner, identity))
end
