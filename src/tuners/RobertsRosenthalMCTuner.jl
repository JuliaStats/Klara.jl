abstract type RobertsRosenthalMCTune <: MCTunerState end

mutable struct UnvRobertsRosenthalMCTune <: RobertsRosenthalMCTune
  logσ::Real # Standard deviation of underlying normal
  δ::Real # Increase or decrease of logσ at each batch
  batch::Integer
  accepted::Integer # Number of accepted MCMC samples during current tuning period
  proposed::Integer # Number of proposed MCMC samples during current tuning period
  totproposed::Integer # Total number of proposed MCMC samples during burnin
  rate::Real # Observed acceptance rate over current tuning period

  function UnvRobertsRosenthalMCTune(
    logσ::Real, δ::Real, batch::Integer, accepted::Integer, proposed::Integer, totproposed::Integer, rate::Real
  )
    @assert batch >= 0 "Batch should be non-negative"
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 <= rate <= 1 "Observed acceptance rate should be in [0, 1]"
    end
    new(logσ, NaN, batch, accepted, proposed, totproposed, rate)
  end
end

UnvRobertsRosenthalMCTune(; logσ::Real=0., accepted::Integer=0, proposed::Integer=0, totproposed::Integer=0) =
  UnvRobertsRosenthalMCTune(logσ, NaN, 0, accepted, proposed, totproposed, NaN)

mutable struct MuvRobertsRosenthalMCTune <: RobertsRosenthalMCTune
  logσ::RealVector # Standard deviation of underlying normal
  δ::Real # Increase or decrease of logσ at each batch
  batch::Integer
  accepted::IntegerVector # Number of accepted MCMC samples during current tuning period
  proposed::Integer # Number of proposed MCMC samples during current tuning period
  totproposed::Integer # Total number of proposed MCMC samples during burnin
  rate::RealVector # Observed acceptance rate over current tuning period

  function MuvRobertsRosenthalMCTune(
    logσ::RealVector,
    δ::Real,
    batch::Integer,
    accepted::IntegerVector,
    proposed::Integer,
    totproposed::Integer,
    rate::RealVector
  )
    @assert batch >= 0 "Batch should be non-negative"
    @assert all(i -> i >= 0, accepted) "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if all(i -> !isnan(i), rate)
      @assert all(i -> 0 <= i <= 1, rate) "Observed acceptance rates should be in [0, 1]"
    end
    new(logσ, NaN, batch, accepted, proposed, totproposed, rate)
  end
end

MuvRobertsRosenthalMCTune(
  logσ::RealVector; accepted::IntegerVector=fill(0, length(logσ)), proposed::Integer=0, totproposed::Integer=0
) =
  MuvRobertsRosenthalMCTune(logσ, NaN, 0, accepted, proposed, totproposed, fill(NaN, length(logσ)))

MuvRobertsRosenthalMCTune(
  d::Integer; logσ::Real=0., accepted::IntegerVector=fill(0, d), proposed::Integer=0, totproposed::Integer=0
) =
  MuvRobertsRosenthalMCTune(fill(logσ, d), NaN, 0, accepted, proposed, totproposed, fill(NaN, d))

reset!(tune::UnvRobertsRosenthalMCTune) = reset_burnin!(tune::MCTunerState)

function reset!(tune::MuvRobertsRosenthalMCTune, i::Integer)
  tune.accepted[i] = 0
  tune.rate[i] = NaN
end

function reset!(tune::MuvRobertsRosenthalMCTune)
  tune.totproposed += tune.proposed
  tune.proposed = 0
end

rate!(tune::MuvRobertsRosenthalMCTune, i::Integer) = (tune.rate[i] = tune.accepted[i]/tune.proposed)

### RobertsRosenthalMCTuner

struct RobertsRosenthalMCTuner <: MCTuner
  targetrate::Real # Target acceptance rate
  period::Integer # Tuning period over which acceptance rate is computed
  verbose::Bool # If verbose=false then the tuner is silent, else it is verbose

  function RobertsRosenthalMCTuner(targetrate::Real, period::Integer, verbose::Bool)
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert period > 0 "Tuning period should be positive"
    new(targetrate, period, verbose)
  end
end

RobertsRosenthalMCTuner(; targetrate::Real=0.44, period::Integer=50, verbose::Bool=false) =
  RobertsRosenthalMCTuner(targetrate, period, verbose)

set_batch!(tune::RobertsRosenthalMCTune, tuner::RobertsRosenthalMCTuner) = (tune.batch = tune.totproposed/tuner.period)

set_delta!(tune::RobertsRosenthalMCTune, tuner::RobertsRosenthalMCTuner) = (tune.δ = min(0.01, tune.batch^-0.5))

tune!(tune::UnvRobertsRosenthalMCTune, tuner::RobertsRosenthalMCTuner) =
  tune.logσ += (tune.rate < tuner.targetrate ? -tune.δ : tune.δ)

tune!(tune::MuvRobertsRosenthalMCTune, tuner::RobertsRosenthalMCTuner, i::Integer) =
  tune.logσ[i] += (tune.rate[i] < tuner.targetrate ? -tune.δ : tune.δ)

show(io::IO, tuner::RobertsRosenthalMCTuner) =
  print(
    io,
    string(
      "RobertsRosenthalMCTuner: target rate = ",
      tuner.targetrate,
      ", period = ",
      tuner.period,
      ", verbose = ",
      tuner.verbose
    )
  )
