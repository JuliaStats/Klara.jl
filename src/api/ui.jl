function run(model::MCModel, sampler::MCSampler, runner::SerialMC, tuner::MCTuner, job::Symbol=:plain)
  if job == :plain
    run(model, sampler, runner, tuner, PlainMCJob(model, sampler, runner, tuner))
  elseif job == :task
    run(model, sampler, runner, tuner, TaskMCJob(model, sampler, runner, tuner))
  end
end
