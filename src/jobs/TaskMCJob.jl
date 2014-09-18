type TaskMCJob <: MCJob
  send::Function
  receive::Function
  task::Task
end

TaskMCJob(task::Task) = MCJob(produce, ()->(), task)
