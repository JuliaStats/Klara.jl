mutable struct DualAveragingMCTune <: MCTunerState
  step::Real # Stepsize of MCMC iteration (for ex leapfrog in HMC or drift stepsize in MALA)
  λ::Real
  μ::Real
  nleaps::Integer
  εbar::Real
  hbar::Real
  hweight::Real
  εweight::Real
  accepted::Integer # Number of accepted MCMC samples during current tuning period
  proposed::Integer # Number of proposed MCMC samples during current tuning period
  totproposed::Integer # Total number of proposed MCMC samples during burnin
  rate::Real # Observed acceptance rate over current tuning period

  function DualAveragingMCTune(
    step::Real,
    λ::Real,
    μ::Real,
    nleaps::Integer,
    εbar::Real,
    hbar::Real,
    hweight::Real,
    εweight::Real,
    accepted::Integer,
    proposed::Integer,
    totproposed::Integer,
    rate::Real
  )
    @assert (step > 0 || isnan(step)) "Stepsize of MCMC iteration should be positive or not a number (NaN)"
    @assert (εbar > 0 || isnan(εbar)) "εbar should be positive or not a number (NaN)"
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 <= rate <= 1 "Observed acceptance rate should be in [0, 1]"
    end
    new(step, λ, μ, nleaps, εbar, hbar, hweight, εweight, accepted, proposed, totproposed, rate)
  end
end

DualAveragingMCTune(;
  step::Real=1.,
  λ::Real=2.,
  εbar::Real=1.,
  hbar::Real=0.,
  accepted::Integer=0,
  proposed::Integer=0,
  totproposed::Integer=0
) =
  DualAveragingMCTune(step, λ, NaN, 0, εbar, hbar, NaN, NaN, accepted, proposed, totproposed, NaN)

struct DualAveragingMCTuner <: MCTuner
  targetrate::Real # Target acceptance rate
  nadapt::Integer
  ε0bar::Real
  h0bar::Real
  γ::Real
  t0::Int
  κ::Real
  period::Integer # Tuning period over which acceptance rate is reported in verbose mode
  verbose::Bool # If verbose=false then the tuner is silent, else it is verbose

  function DualAveragingMCTuner(
    targetrate::Real,
    nadapt::Integer,
    ε0bar::Real,
    h0bar::Real,
    γ::Real,
    t0::Int,
    κ::Real,
    period::Integer,
    verbose::Bool
  )
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert nadapt > 0 "Number of adaptation steps should be positive"
    @assert ε0bar > 0 "ε0bar should be positive"
    @assert period > 0 "Period over which acceptance rate is reported in verbose mode should be positive"
    @assert t0 > 0 "t0 should be positive"
    new(targetrate, nadapt, ε0bar, h0bar, γ, t0, κ, period, verbose)
  end
end

DualAveragingMCTuner(
  targetrate::Real,
  nadapt::Integer;
  ε0bar::Real=1.,
  h0bar::Real=0.,
  γ::Real=0.05,
  t0::Integer=10,
  κ::Real=0.75,
  period::Integer=100,
  verbose::Bool=false
) = DualAveragingMCTuner(targetrate, nadapt, ε0bar, h0bar, γ, t0, κ, period, verbose)

function tune!(tune::DualAveragingMCTune, tuner::DualAveragingMCTuner, count::Integer, a::Real)
  tune.hweight = 1/(count+tuner.t0)
  tune.hbar = (1-tune.hweight)*tune.hbar+tune.hweight*(tuner.targetrate-a)
  tune.step = exp(tune.μ-sqrt(count)*tune.hbar/tuner.γ)
  tune.εweight = count^(-tuner.κ)
  tune.εbar = exp((1-tune.εweight)*log(tune.εbar)+tune.εweight*log(tune.step))
end

show(io::IO, tuner::DualAveragingMCTuner) =
  print(
    io,
    string(
      "DualAveragingMCTuner: target rate = ",
      tuner.targetrate,
      ", nadapt = ",
      tuner.nadapt,
      ", ε0bar = ",
      tuner.ε0bar,
      ", h0bar = ",
      tuner.h0bar,
      ", γ = ",
      tuner.γ,
      ", t0 = ",
      tuner.t0,
      ", κ = ",
      tuner.κ,
      ", period = ",
      tuner.period,
      ", verbose = ",
      tuner.verbose
    )
  )
