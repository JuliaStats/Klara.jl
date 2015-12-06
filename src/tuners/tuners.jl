### MCTunerState subtypes hold the samplers' temporary output used for tuning the sampler

abstract MCTunerState

function reset_burnin!(tune::MCTunerState)
  tune.totproposed += tune.proposed
  (tune.accepted, tune.proposed, tune.rate) = (0, 0, NaN)
end

rate!(tune::MCTunerState) = (tune.rate = tune.accepted/tune.proposed)

### Root Monte Carlo Tuner

abstract MCTuner
