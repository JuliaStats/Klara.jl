### COnvenience functions for constructing Monte Carlo jobs

function initialize_mcjob(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner=VanillaMCTuner(), job::Symbol=:task)
  mcjob::MCJob
  if job == :plain
    mcjob = PlainMCJob(m, s, r, t)
  elseif job == :task
    mcjob = TaskMCJob(m, s, r, t)
  end
  mcjob
end

function initialize_mcjob(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner, job::MCJob)
  mcjob::MCJob
  if isa(job, PlainMCJob)
    mcjob = initialize_mcjob(m, s, r, t, :plain)
  elseif isa(job, TaskMCJob)
    mcjob = initialize_mcjob(m, s, r, t, :task)
  end
  mcjob
end
