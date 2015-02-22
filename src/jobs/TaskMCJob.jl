### TaskMCJob type is used for Monte Carlo jobs based on coroutines (tasks)

type TaskMCJob{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner} <: MCJob
  model::Vector{M}
  sampler::Vector{S}
  runner::Vector{R}
  tuner::Vector{T}
  heap::Vector{MCHeap}
  send::Function
  receive::Function
  reset::Function
  jobtype::Symbol
  dim::Int
  task::Vector{Task}

  function TaskMCJob(m::Vector{M}, s::Vector{S}, r::Vector{R}, t::Vector{T})
    job::TaskMCJob = new()

    job.dim = length(m)
    @assert job.dim == length(s) == length(r) == length(t) "Number of models, samplers, runners and tuners not equal."

    job.model, job.sampler, job.runner, job.tuner = m, s, r, t
    job.heap = MCHeap[initialize_heap(m[i], s[i], r[i], t[i]) for i = 1:job.dim]
    job.task = Task[Task(()->initialize_task!(job.heap[i], m[i], s[i], r[i], t[i])) for i = 1:job.dim]
    job.send = produce
    job.receive = (i::Int)->consume(job.task[i])
    job.reset = (i::Int, x::Vector{Float64})->reset(job.task[i], x)
    job.jobtype = :task

    job
  end
end

TaskMCJob{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner}(m::M, s::S, r::R, t::T) =
  TaskMCJob{M, S, R, T}([m], [s], [r], [t])
