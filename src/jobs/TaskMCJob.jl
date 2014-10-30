### TaskMCJob type is used for Monte Carlo jobs based on coroutines (tasks)

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
