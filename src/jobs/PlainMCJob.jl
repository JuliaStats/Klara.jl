### Plain Monte Carlo jobs do not use tasks

type PlainMCJob <: MCJob
  send::Function
  receive::Function
end

PlainMCJob() = PlainMCJob(identity, ()->())

function PlainMCJob(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner)
  stash::MCStash = initialize(m, s, r, t)
  PlainMCJob(identity, ()->iterate!(stash, m, s, r, t, identity))
end
