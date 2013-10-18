import Base.run
export run, resume, prun

stop!(c::MCMCChain) = (c.task.task.done == false ? c.task.task.done = true : nothing)

# General run() function which invoke run function specific to Task.runner field
function run(t::MCMCTask)
  if isa(t.runner, SerialMC)
    run_serialmc(t)
  end
end

# Chain continuation alternate
run(c::MCMCChain) = run(c.task)

# vectorized version of 'run' for arrays of MCMCTasks or MCMCChains
function run(t::Array{MCMCTask}; args...)
  lastrunner = t[end].runner
  @assert all(map(t->isa(t.runner, typeof(lastrunner)), t)) "Runners do not have the same runner type"

  if isa(lastrunner, SerialMC)
    res = Array(MCMCChain, size(t))
    for i = 1:length(t)
      res[i] = run(t[i])
    end 
    res
  elseif isa(lastrunner, SerialTempMC)
    run_serialtempmc(t)
  else isa(lastrunner, SeqMC)
    run_seqmc(t; args...)    
  end
end

# parallel vectorized version of 'run' for arrays of MCMCTasks or MCMCChains
function prun(t::Array{MCMCTask}; args...)
  lastrunner = t[end].runner
  @assert all(map(t->isa(t.runner, typeof(lastrunner)), t)) "Runners do not have the same runner type"

  if isa(lastrunner, SerialMC)
    pmap(run_serialmc_exit, t)   
  end
end

# Alternate version with model, sampler and runner passed separately
run{M<:MCMCModel, S<:MCMCSampler}(m::Union(M, Vector{M}), s::Union(S, Vector{S}), r::MCMCRunner) = run(m*s*r)

# Functions for resuming MCMCTasks as well as arrays of MCMCTasks
function resume(t::MCMCTask; steps::Int=100)
  if isa(t.runner, SerialMC)
    resume_serialmc(t, steps=steps)
  end
end

resume(c::MCMCChain; steps::Int=100) = resume(c.task, steps=steps)

function resume(t::Array{MCMCTask}; steps::Int=100, args...)
  if isa(t[end].runner, SerialMC)
    res = Array(MCMCChain, size(t))
      for i = 1:length(t)
        res[i] = resume(t[i], steps=steps)
      end 
    res
  elseif isa(t[end].runner, SerialTempMC)
    resume_serialtempmc(t, steps=steps)
  else isa(t[end].runner, SeqMC)
    resume_seqmc(t; steps=steps, args...)    
  end
end
