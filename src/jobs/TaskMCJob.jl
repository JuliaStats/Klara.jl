### TaskMCJob type is used for Monte Carlo jobs based on coroutines (tasks)

type TaskMCJob <: MCJob
  send::Function
  receive::Function
  task::Task
end

TaskMCJob(task::Task) = TaskMCJob(produce, ()->consume(task), task)
TaskMCJob() = TaskMCJob(produce, ()->(), Task(()->()))

function TaskMCJob(model::MCModel, sampler::HMCSampler, runner::SerialMC, tuner::MCTuner)
  task::Task = initialize_task(model, sampler, runner, tuner)
  TaskMCJob(produce, ()->consume(task), task)
end
