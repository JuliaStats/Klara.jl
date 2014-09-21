### TaskMCJob type is used for Monte Carlo jobs based on coroutines (tasks)

type TaskMCJob <: MCJob
  send::Function
  receive::Function
  task::Task
end

TaskMCJob(t::Task) = TaskMCJob(produce, ()->consume(t), t)
TaskMCJob() = TaskMCJob(produce, ()->(), Task(()->()))

function TaskMCJob(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner)
  mctask::Task = Task(()->initialize_task(m, s, r, t))
  TaskMCJob(produce, ()->consume(mctask), mctask)
end
