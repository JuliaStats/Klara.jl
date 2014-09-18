### Vanilla MCMC is a dummy tuner type in the sense that it does not perform any tuning
### It is used only for determining whether the MCMC sampler will be verbose

immutable VanillaMCTuner <: MCTuner
  period::Int # Period over which acceptance rate is computed
  verbose::Bool # Specify whether the tuner will be in verbose or silent mode

  function VanillaMCTuner(period::Int, verbose::Bool)
    @assert period > 0 "Adaptation period should be positive."
    new(period, verbose)
  end
end

VanillaMCTuner(; period::Int=100, verbose::Bool=false) = VanillaMCTuner(period, verbose)
