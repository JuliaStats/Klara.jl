### Plain Monte Carlo jobs do not use tasks

type PlainMCJob <: MCJob
  model::MCModel
  sampler::MCSampler
  runner::MCRunner
  tuner::MCTuner
  stash::MCStash
  send::Function
  receive::Function
  reset::Function
  jtype::Symbol

  function PlainMCJob(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner)
    job::PlainMCJob = new()
    job.model, job.sampler, job.runner, job.tuner = m, s, r, t
    job.stash = initialize_stash(m, s, r, t)
    job.send = identity
    job.receive = ()->iterate!(job.stash, m, s, r, t, identity)
    job.reset = (x::Vector{Float64})->reset!(job.stash, x)
    job.jtype = :plain
    job
  end
end
