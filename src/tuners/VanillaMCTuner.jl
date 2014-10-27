### VanillaMCTuner is a dummy tuner type in the sense that it does not perform any tuning
### It is used only for determining whether the Monte Carlo sampler will be verbose

immutable VanillaMCTuner <: MCTuner
  period::Int # Period over which acceptance rate is computed
  verbose::Bool # Specify whether the tuner will be in verbose or silent mode

  function VanillaMCTuner(period::Int, verbose::Bool)
    @assert period > 0 "Adaptation period should be positive."
    new(period, verbose)
  end
end

VanillaMCTuner(; period::Int=100, verbose::Bool=false) = VanillaMCTuner(period, verbose)

### VanillaMCTune holds the temporary output of a Monte Carlo sampler that uses the VanillaMCMCTuner

type VanillaMCTune <: MCTune
  accepted::Int # Number of accepted Monte Carlo samples within current tuner period
  proposed::Int # Number of proposed Monte Carlo samples within current tuner period
  rate::Float64 # Acceptance rate over current tuner period

  function VanillaMCTune(accepted::Int, proposed::Int, rate::Float64)
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative."
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative."
    new(accepted, proposed, rate)
  end
end

VanillaMCTune(accepted::Int, proposed::Int) = VanillaMCTune(accepted::Int, proposed::Int, NaN)
VanillaMCTune() = VanillaMCTune(0, 0, NaN)

reset!(tune::VanillaMCTune) = ((tune.accepted, tune.proposed) = (0, 0))

count!(tune::VanillaMCTune) = (tune.accepted += 1)

rate!(tune::VanillaMCTune) = (tune.rate = tune.accepted/tune.proposed)
