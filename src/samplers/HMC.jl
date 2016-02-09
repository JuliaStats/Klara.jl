### Abstract HMC state

abstract HMCState <: MCSamplerState

### HMC state subtypes

## MuvHMCState holds the internal state ("local variables") of the HMC sampler for multivariate parameters

type MuvHMCState{N<:Real} <: HMCState
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by HMC
  leapstep::Real # Leapfrog stepsize for a single Monte Carlo iteration
  tune::MCTunerState
  ratio::Real
  oldmomentum::Vector{N}
  newmomentum::Vector{N}
  oldhamiltonian::Real
  newhamiltonian::Real

  function MuvHMCState(
    pstate::ParameterState{Continuous, Multivariate},
    leapstep::Real,
    tune::MCTunerState,
    ratio::Real,
    oldmomentum::Vector{N},
    newmomentum::Vector{N},
    oldhamiltonian::Real,
    newhamiltonian::Real
  )
    if !isnan(leapstep)
      @assert leapstep > 0 "Drift step size must be positive"
    end
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(pstate, leapstep, tune, ratio, oldmomentum, newmomentum, oldhamiltonian, newhamiltonian)
  end
end
