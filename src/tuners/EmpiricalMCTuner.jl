### Empirical Monte Carlo tuner

immutable EmpiricalMCTuner <: MCTuner
  targetrate::Float64 # Target acceptance rate
  period::Int # Adaptation period, which determines when to attempt tuning
  adaptnsteps::Bool # Determine whether to adapt the number of steps along with stepsize
  maxnsteps::Int # Maximum number of steps allowed (for ex maximum number of leapfrog steps)
  targetlen::Float64 # Target length (this can be thought of as adaptation step size times number of steps)
  verbose::Bool # Specify whether the tuner will be in verbose or silent mode

  function EmpiricalMCTuner(targetrate::Float64, period::Int, adaptnsteps::Bool, maxnsteps::Int, targetlen::Float64,
    verbose::Bool)
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1."
    @assert period > 0 "Adaptation period should be positive."
    if adaptnsteps
      @assert maxnsteps > 0 "Maximum number of steps should be positive."
    end
    new(targetrate, period, maxnsteps, targetlen, verbose)
  end
end

EmpiricalMCTuner(targetrate::Float64; period::Int=100, adaptnsteps::Bool=true, maxnsteps::Int=200,
  targetlen::Float64=1., verbose::Bool=false) =
  EmpiricalMCTuner(targetrate, period, adaptnsteps, maxnsteps, targetlen, verbose)

### EmpiricalMCTune holds the temporary output of a Monte Carlo sampler that uses the EmpiricalMCMCTuner

type EmpiricalMCTune <: MCTune
  step::Float64 # Stepsize of current Monte Carlo iteration (for ex leapfrog or drift stepsize)
  nsteps::Int # Number of steps of current Monte Carlo iteration (for ex number of leapfrog steps)
  accepted::Int # Number of accepted Monte Carlo samples within current tuner period
  proposed::Int # Number of proposed Monte Carlo samples within current tuner period
  rate::Float64 # Acceptance rate over current tuner period

  function EmpiricalMCTune(step::Float64, nsteps::Int, accepted::Int, proposed::Int, rate::Float64)
    @assert step > 0 "Stepsize of current MCMC iteration should be positive."
    @assert nsteps > 0 "Number of steps of current MCMC iteration should be positive."
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative."
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative."
    new(step, nsteps, accepted, proposed, rate)
  end
end

EmpiricalMCTune(step::Float64, nsteps::Int, accepted::Int, proposed::Int) =
  EmpiricalMCTune(step::Float64, nsteps::Int, accepted::Int, proposed::Int, NaN)
EmpiricalMCTune(step::Float64, nsteps::Int) = EmpiricalMCTune(step::Float64, nsteps::Int, 0, 0, NaN)
EmpiricalMCTune(step::Float64) = EmpiricalMCTune(step::Float64, 10, 0, 0, NaN)

reset!(tune::EmpiricalMCTune) = ((tune.accepted, tune.proposed) = (0, 0))

count!(tune::EmpiricalMCTune) = (tune.accepted += 1)

function adapt!(tune::EmpiricalMCTune, tuner::EmpiricalMCTuner)
  tune.rate = tune.accepted/tune.proposed
  tune.step *= (1/(1+exp(-11*(tune.rate-tuner.targetrate)))+0.5)
  if tuner.adaptnsteps
    tune.nsteps = min(tuner.maxnsteps, ceil(tuner.targetlen/tune.step))
  end
end
