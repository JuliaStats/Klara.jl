### MCTunerState subtypes hold the samplers' temporary output used for tuning the sampler

abstract type MCTunerState end

mutable struct BasicMCTune <: MCTunerState
  step::Real # Stepsize of MCMC iteration (for ex leapfrog in HMC or drift stepsize in MALA)
  accepted::Integer # Number of accepted MCMC samples during current tuning period
  proposed::Integer # Number of proposed MCMC samples during current tuning period
  totproposed::Integer # Total number of proposed MCMC samples during burnin
  rate::Real # Observed acceptance rate over current tuning period

  function BasicMCTune(step::Real, accepted::Integer, proposed::Integer, totproposed::Integer, rate::Real)
    @assert (step > 0 || isnan(step)) "Stepsize of MCMC iteration should be positive or not a number (NaN)"
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 <= rate <= 1 "Observed acceptance rate should be in [0, 1]"
    end
    new(step, accepted, proposed, totproposed, rate)
  end
end

BasicMCTune(step::Real=1., accepted::Integer=0, proposed::Integer=0, totproposed::Integer=0) =
  BasicMCTune(step, accepted, proposed, totproposed, NaN)

function reset_burnin!(tune::MCTunerState)
  tune.totproposed += tune.proposed
  (tune.accepted, tune.proposed, tune.rate) = (0, 0, NaN)
end

rate!(tune::MCTunerState) = (tune.rate = tune.accepted/tune.proposed)

### Root Monte Carlo Tuner

abstract type MCTuner end
