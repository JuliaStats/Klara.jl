function MCJob(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner=VanillaMCTuner(), j::Symbol=:task)
  mcjob::MCJob
  if j == :plain
    mcjob = PlainMCJob(m, s, r, t)
  elseif j == :task
    mcjob = TaskMCJob(m, s, r, t)
  end
  mcjob
end

# Convenience constructors for setting up a vector of Monte Carlo jobs

MCJob(m::MCModel, s::MCSampler, r::MCRunner, t::MCTuner=VanillaMCTuner()) = MCJob(m, s, r, t)
MCJob{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner}(m::Vector{M}, s::Vector{S}, r::Vector{R}, t::Vector{T}) =
  map(MCJob, m, s, r, t)
MCJob{M<:MCModel, S<:MCSampler, R<:MCRunner}(m::Vector{M}, s::Vector{S}, r::Vector{R}) = map(MCJob, m, s, r)

run(j::MCJob) = run(j.model, j.sampler, j.runner, j.tuner, j.jtype)

run{J<:MCJob}(j::Vector{J}) = map(run, j)

resume!(c::MCChain, j::MCJob; nsteps::Int=100) =
  resume!(c, j.model, j.sampler, j.runner, j.tuner, j.jtype; nsteps=nsteps)

resume(c::MCChain, j::MCJob; nsteps::Int=100) =
  resume!(c, deepcopy(j.model), j.sampler, j.runner, j.tuner, j.jtype; nsteps=nsteps)
