### TaskMCJob type is used for Monte Carlo jobs based on coroutines (tasks)

type TaskMCJob <: MCJob
  model::MCModel
  sampler::MCSampler
  runner::MCRunner
  tuner::MCTuner
  stash::MCStash
  send::Function
  receive::Function
  reset::Function
  jtype::Symbol
  task::Task

  function TaskMCJob(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner)
    job::TaskMCJob = new()
    job.model, job.sampler, job.runner, job.tuner = m, s, r, t
    job.stash = initialize(m, s, r, t)
    job.task = Task(()->initialize_task!(job.stash, m, s, r, t))
    job.send = produce
    job.receive = ()->consume(job.task)
    job.reset = (x::Vector{Float64})->reset(job.task, x)
    job.jtype = :task
    job
  end
end
