### Plain Monte Carlo jobs do not use tasks

# The runner field is not of Vector{MCRunner} type because the coded (and perhaps all existing) serial and sequential
# Monte Carlo algorithms presume common burnin, thinning and total number of steps between the simulated chains. If it
# is ever needed to have chain-specific runners, then the runner field can be turned to a vector of runners.
type PlainMCJob{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner} <: MCJob
  model::Vector{M}
  sampler::Vector{S}
  runner::R
  tuner::Vector{T}
  stash::Vector{MCStash}
  send::Function
  receive::Function
  reset::Function
  jtype::Symbol

  function PlainMCJob(m::Vector{M}, s::Vector{S}, r::R, t::Vector{T})
    nchains = length(m)
    @assert nchains == length(s) == length(t) "Number of models, samplers and tuners not equal."

    job::PlainMCJob = new()

    job.model, job.sampler, job.runner, job.tuner = m, s, r, t
    job.stash = MCStash[initialize_stash(m[i], s[i], r, t[i]) for i = 1:nchains]
    job.send = identity
    job.receive = (i::Int)->iterate!(job.stash[i], m, s, r, t, identity)
    job.reset = (i::Int, x::Vector{Float64})->reset!(job.stash[i], x)
    job.jtype = :plain

    job
  end
end

PlainMCJob{M<:MCModel, S<:MCSampler, T<:MCTuner}(m::M, s::S, r::SerialMC, t::T) =
  PlainMCJob{M, S, SerialMC, T}([m], [s], r, [t])
