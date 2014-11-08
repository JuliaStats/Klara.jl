# MCState holds two samples at each Monte Carlo iteration, the current and the successive one

type MCState{S<:MCSample}
  successive::S # If proposed sample is accepted, then successive = proposed, otherwise successive = current
  current::S
  diagnostics::Dict
end

MCState{S<:MCSample}(p::S, c::S) = MCState(p, c, Dict())
