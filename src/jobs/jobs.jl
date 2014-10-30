### Constructors for setting up a Monte Carlo job

function MCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::M, s::S, r::SerialMC, t::T, j::Symbol)
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

MCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::M, s::S, r::SerialMC; tuner::T=VanillaMCTuner(), job::Symbol=:task) =
  MCJob(m, s, r, tuner, job)

### Constructors for setting up a vector of Monte Carlo jobs

function MCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::Vector{M}, s::Vector{S}, r::Vector{SerialMC}, t::Vector{T},
  j::Vector{Symbol})
  @assert length(m) == length(s) == length(r) == length(t) == length(j)
    "Number of models, samplers, runners, tuners and job types not equal."
  map(MCJob, m, s, r, t, j)
end

MCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::Vector{M}, s::Vector{S}, r::Vector{SerialMC};
  tuners::Vector{T}=fill(VanillaMCTuner(), length(m)), jobs::Vector{Symbol}=fill(:task, length(m))) =
  map(MCJob, m, s, r, tuners, jobs)

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
    resume!(j.model[1], c, j.sampler[1], j.runner, j.tuner[1], j.jtype; nsteps=nsteps)
  end
end

function resume(j::MCJob, c::MCChain; nsteps::Int=100)
  if isa(j.runner, SerialMC)
    resume!(deepcopy(j.model)[1], c, j.sampler[1], j.runner, j.tuner[1], j.jtype; nsteps=nsteps)
  end
end
