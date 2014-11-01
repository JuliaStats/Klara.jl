### Plain Monte Carlo jobs do not use tasks

type PlainMCJob{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner} <: MCJob
  model::Vector{M}
  sampler::Vector{S}
  runner::Vector{R}
  tuner::Vector{T}
  stash::Vector{MCStash}
  send::Function
  receive::Function
  reset::Function
  jobtype::Symbol
  dim::Int

  function PlainMCJob(m::Vector{M}, s::Vector{S}, r::Vector{R}, t::Vector{T})
    job::PlainMCJob = new()

    job.dim = length(m)
    @assert job.dim == length(s) == length(r) == length(t) "Number of models, samplers, runners and tuners not equal."

    job.model, job.sampler, job.runner, job.tuner = m, s, r, t
    job.stash = MCStash[initialize_stash(m[i], s[i], r[i], t[i]) for i = 1:job.dim]
    job.send = identity
    job.receive = (i::Int)->iterate!(job.stash[i], m[i], s[i], r[i], t[i], identity)
    job.reset = (i::Int, x::Vector{Float64})->reset!(job.stash[i], x)
    job.jobtype = :plain

    job
  end
end

PlainMCJob{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner}(m::M, s::S, r::R, t::T) =
  PlainMCJob{M, S, R, T}([m], [s], [r], [t])
