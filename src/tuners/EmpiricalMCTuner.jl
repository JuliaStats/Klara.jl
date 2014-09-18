### Empirical MCMC tuner

immutable EmpiricalMCTuner <: MCTuner
  targetrate::Float64 # Target acceptance rate
  period::Int # Adaptation period, which determines when to attempt tuning
  adaptnsteps::Bool # Determine whether to adapt the number of steps along with stepsize
  maxnsteps::Int # Maximum number of steps allowed (for ex maximum number of leapfrog or drift steps)
  targetlen::Float64 # Target length (this can be thought of as adaptation step size times number of steps)
  verbose::Bool # Specify whether the tuner will be in verbose or silent mode

  function EmpiricalMCTuner(targetrate::Float64, period::Int, adaptnsteps::Bool, maxnsteps::Int, targetlen::Float64,
    verbose::Bool)
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1."
    @assert period > 0 "Adaptation period should be positive."
    @assert adaptnsteps == true && maxnsteps > 0 "Maximum number of steps should be positive."
    new(targetrate, period, maxnsteps, targetlen, verbose)
  end
end

EmpiricalMCTuner(targetrate::Float64; period::Int=100, adaptnsteps::Bool=true, maxnsteps::Int=200,
  targetlen::Float64=1., verbose::Bool=false) =
  EmpiricalMCTuner(targetrate, period, adaptnsteps, maxnsteps, targetlen, verbose)
