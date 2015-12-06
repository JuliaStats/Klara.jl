### Auxiliary functions used as scores for penalising deviation of observed from target acceptance rate

## logistic_rate_score

# logistic_rate_score uses the logistic function to scale the acceptance rate by a factor ranging from 0 to 2
# In other words, it allows to nearly eliminate or double the rate depending on its observed value
# k gives the curve's steepness. For larger k, the curve becomes more steep

logistic_rate_score(x::Real, k::Real=7.) = logistic(x, 2., k, 0., 0.)

## erf_rate_score

# erf_rate_score uses the error function (erf) to scale the acceptance rate by a factor ranging from 0 to 2
# In other words, it allows to nearly eliminate or double the rate depending on its observed value
# k gives the curve's steepness. For larger k, the curve becomes more steep

erf_rate_score(x::Real, k::Real=3.) = erf(k*x)+1

### AcceptanceRateMCTune

# AcceptanceRateMCTune holds the tuning-related local variables of a MCSampler that uses the AcceptanceRateMCTuner

type AcceptanceRateMCTune <: MCTunerState
  step::Real # Stepsize of MCMC iteration (for ex leapfrog in HMC or drift stepsize in MALA)
  accepted::Int # Number of accepted MCMC samples during current tuning period
  proposed::Int # Number of proposed MCMC samples during current tuning period
  totproposed::Int # Total number of proposed MCMC samples during burnin
  rate::Real # Observed acceptance rate over current tuning period

  function AcceptanceRateMCTune(step::Real, accepted::Int, proposed::Int, totproposed::Int, rate::Real)
    @assert step > 0 "Stepsize of MCMC iteration should be positive"
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 < rate < 1 "Observed acceptance rate should be between 0 and 1"
    end
    new(step, accepted, proposed, totproposed, rate)
  end
end

AcceptanceRateMCTune(step::Real=1., accepted::Int=0, proposed::Int=0, totproposed::Int=0) =
  AcceptanceRateMCTune(step, accepted, proposed, totproposed, NaN)

### AcceptanceRateMCTuner

# AcceptanceRateMCTuner tunes empirically on the basis of the discrepancy between observed and target acceptance rate
# This discrepancy is pernalised via a score function set by the user
# The default score function is a stretched logistic map

immutable AcceptanceRateMCTuner <: MCTuner
  targetrate::Real # Target acceptance rate
  score::Function # Score function for penalising discrepancy between observed and target acceptance rate
  period::Int # Tuning period over which acceptance rate is computed
  verbose::Bool # If verbose=false then the tuner is silent, else it is verbose

  function AcceptanceRateMCTuner(targetrate::Real, score::Function, period::Int, verbose::Bool)
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert period > 0 "Tuning period should be positive"
    new(targetrate, score, period, verbose)
  end
end

AcceptanceRateMCTuner(
  targetrate::Real;
  score::Function=logistic_rate_score,
  period::Int=100,
  verbose::Bool=false
) =
  AcceptanceRateMCTuner(targetrate, score, period, verbose)

tune!(tune::AcceptanceRateMCTune, tuner::AcceptanceRateMCTuner) = (tune.step *= tuner.score(tune.rate-tuner.targetrate))
