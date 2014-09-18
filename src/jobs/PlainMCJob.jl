type PlainMCJob <: MCJob
  send::Function
  receive::Function
end

PlainMCJob() = MCJob(identity, ()->())
