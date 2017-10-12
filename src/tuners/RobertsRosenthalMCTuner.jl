abstract type RobertsRosenthalMCTune{F<:VariateForm} <: MCTunerState end

mutable struct UnvRobertsRosenthalMCTune <: RobertsRosenthalMCTune{Univariate}
  logσ::Real # Standard deviation of underlying normal
  δ::Real # Increase or decrease of logσ at each batch
  count::Integer
  batch::Integer
  accepted::Integer # Number of accepted MCMC samples during current tuning period
  proposed::Integer # Number of proposed MCMC samples during current tuning period
  totproposed::Integer # Total number of proposed MCMC samples during burnin
  rate::Real # Observed acceptance rate over current tuning period

  function UnvRobertsRosenthalMCTune(
    logσ::Real,
    δ::Real,
    count::Integer,
    batch::Integer,
    accepted::Integer,
    proposed::Integer,
    totproposed::Integer,
    rate::Real
  )
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    @assert batch >= 0 "Batch should be non-negative"
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 <= rate <= 1 "Observed acceptance rate should be in [0, 1]"
    end
    new(logσ, NaN, count, batch, accepted, proposed, totproposed, rate)
  end
end

UnvRobertsRosenthalMCTune(logσ::Real=0., accepted::Integer=0, proposed::Integer=0, totproposed::Integer=0) =
  UnvRobertsRosenthalMCTune(σ, 0, 0, accepted, proposed, totproposed, NaN)

mutable struct MuvRobertsRosenthalMCTune <: RobertsRosenthalMCTune{Multivariate}
  logσ::RealVector # Standard deviation of underlying normal
  δ::Real # Increase or decrease of logσ at each batch
  count::Integer
  batch::Integer
  accepted::IntegerVector # Number of accepted MCMC samples during current tuning period
  proposed::Integer # Number of proposed MCMC samples during current tuning period
  totproposed::Integer # Total number of proposed MCMC samples during burnin
  rate::RealVector # Observed acceptance rate over current tuning period

  function MuvRobertsRosenthalMCTune(
    logσ::RealVector,
    δ::Real,
    count::Integer,
    batch::Integer,
    accepted::IntegerVector,
    proposed::Integer,
    totproposed::Integer,
    rate::RealVector
  )
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    @assert batch >= 0 "Batch should be non-negative"
    @assert all(i -> i >= 0, accepted) "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if all(i -> !isnan(i), rate)
      @assert all(i -> 0 <= i <= 1, rate) "Observed acceptance rates should be in [0, 1]"
    end
    new(logσ, NaN, count, batch, accepted, proposed, totproposed, rate)
  end
end

function MuvRobertsRosenthalMCTune(logσ::RealVector, proposed::Integer=0, totproposed::Integer=0)
  l = length(logσ)
  MuvRobertsRosenthalMCTune(logσ, NaN, 0, 0, fill(0, l), proposed, totproposed, fill(NaN, l))
end

MuvRobertsRosenthalMCTune(size::Integer, proposed::Integer=0, totproposed::Integer=0) =
  MuvRobertsRosenthalMCTune(fill(0., size), NaN, 0, 0, fill(0, size), proposed, totproposed, fill(NaN, size))

### RobertsRosenthalMCTuner

struct RobertsRosenthalMCTuner <: MCTuner
  targetrate::Real # Target acceptance rate
  nbatch::Integer # Number of batches
  period::Integer # Tuning period over which acceptance rate is computed
  verbose::Bool # If verbose=false then the tuner is silent, else it is verbose

  function RobertsRosenthalMCTuner(targetrate::Real, nadapt::Integer, period::Integer, verbose::Bool)
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert nbatch > 0 "Number of batches should be positive"
    @assert period > 0 "Tuning period should be positive"
    new(targetrate, nbatch, period, verbose)
  end
end

RobertsRosenthalMCTuner(
  targetrate::Real,
  nbatch::Integer;
  period::Integer=100,
  verbose::Bool=false
) =
  RobertsRosenthalMCTuner(targetrate, nbatch, period, verbose)

set_batch!(tune::RobertsRosenthalMCTune, tuner::RobertsRosenthalMCTuner) = (tune.batch = tune.count/tuner.nbatch)

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
      ", nbatch = ",
      tuner.nbatch,
      ", period = ",
      tuner.period,
      ", verbose = ",
      tuner.verbose
    )
  )
