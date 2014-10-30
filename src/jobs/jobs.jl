### Constructors for setting up a Monte Carlo job

function MCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::M, s::S, r::SerialMC, t::T=VanillaMCTuner(), j::Symbol=:task)
  mcjob::MCJob
  if j == :plain
    mcjob = PlainMCJob(m, s, r, t)
  elseif j == :task
    mcjob = TaskMCJob(m, s, r, t)
  else
    error("Only :plain and :task jobs are available.")
  end
  mcjob
end

### Constructors for setting up a vector of Monte Carlo jobs

MCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::Vector{M}, s::Vector{S}, r::Vector{SerialMC},
  t::Vector{T}=fill(VanillaMCTuner(), length(m)), j::Vector{Symbol}=fill(:task, length(m))) =
  map(MCJob, m, s, r, t)

### Functions for running jobs

function run(j::MCJob)
  if isa(j.runner, SerialMC)
    run(j.model[1], j.sampler[1], j.runner, j.tuner[1], j.jtype)
  end
end

run{J<:MCJob}(j::Vector{J}) = map(run, j)

### Functions for resuming jobs

function resume!(j::MCJob, c::MCChain; nsteps::Int=100)
  if isa(j.runner, SerialMC)  
    resume!(c, j.model, j.sampler, j.runner, j.tuner, j.jtype; nsteps=nsteps)
  end
end

function resume(j::MCJob, c::MCChain; nsteps::Int=100)
  if isa(j.runner, SerialMC)
    resume!(c, deepcopy(j.model), j.sampler, j.runner, j.tuner, j.jtype; nsteps=nsteps)
  end
end
