### TaskMCJob type is used for Monte Carlo jobs based on coroutines (tasks)

# The runner field is not of Vector{MCRunner} type because the coded (and perhaps all existing) serial and sequential
# Monte Carlo algorithms presume common burnin, thinning and total number of steps between the simulated chains. If it
# is ever needed to have chain-specific runners, then the runner field can be turned to a vector of runners.
type TaskMCJob{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner} <: MCJob
  model::Vector{M}
  sampler::Vector{S}
  runner::R
  tuner::Vector{T}
  stash::Vector{MCStash}
  send::Function
  receive::Function
  reset::Function
  jtype::Symbol
  task::Vector{Task}

  function TaskMCJob(m::Vector{M}, s::Vector{S}, r::R, t::Vector{T})
    nchains = length(m)
    @assert nchains == length(s) == length(t) "Number of models, samplers and tuners not equal."

    job::TaskMCJob = new()

    job.model, job.sampler, job.runner, job.tuner = m, s, r, t
    job.stash = MCStash[initialize_stash(m[i], s[i], r, t[i]) for i = 1:nchains]
    job.task = Task[Task(()->initialize_task!(job.stash[i], m[i], s[i], r, t[i])) for i = 1:nchains]
    job.send = produce
    job.receive = (i::Int)->consume(job.task[i])
    job.reset = (i::Int, x::Vector{Float64})->reset(job.task[i], x)
    job.jtype = :task

    job
  end
end

TaskMCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::M, s::S, r::SerialMC, t::T) =
  TaskMCJob{M, S, SerialMC, T}([m], [s], r, [t])
