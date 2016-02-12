### ARSState

# ARSState holds the internal state ("local variables") of the acceptance-rejection sampler

type ARSState <: MCSamplerState
  pstate::ParameterState # Parameter state used internally by ARS
  tune::MCTunerState
  logproposal::Real
  weight::Real
end

ARSState(pstate::ParameterState, tune::MCTunerState=VanillaMCTune()) = ARSState(pstate, tune, NaN, NaN)

### Acceptance-rejection sampler (ARS)

immutable ARS <: MCSampler
  logproposal::Function # Possibly unnormalized log-proposal (envelope)
  proposalscale::Real # Scale factor to ensure scaled-up logproposal (envelope) covers target
  jumpscale::Real # Scale factor for adapting jump size

  function ARS(logproposal::Function, proposalscale::Real, jumpscale::Real)
    @assert jumpscale > 0 "Scale factor for adapting jump size is not positive"
    new(logproposal, proposalscale, jumpscale)
  end
end

ARS(logproposal::Function; proposalscale::Real=1., jumpscale::Real=1.) = ARS(logproposal, proposalscale, jumpscale)
