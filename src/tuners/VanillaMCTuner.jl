### VanillaMCTuner

# VanillaMCTuner is a dummy tuner type in the sense that it does not perform any tuning
# It is used only for determining whether the MCSampler will be verbose

struct VanillaMCTuner <: MCTuner
  period::Integer # Tuning period over which acceptance rate is computed
  verbose::Bool # If verbose=false then the tuner is silent, else it is verbose

  function VanillaMCTuner(period::Integer, verbose::Bool)
    @assert period > 0 "Adaptation period should be positive"
    new(period, verbose)
  end
end

VanillaMCTuner(; period::Integer=100, verbose::Bool=false) = VanillaMCTuner(period, verbose)

show(io::IO, tuner::VanillaMCTuner) =
  print(io, "VanillaMCTuner: period = $(tuner.period), verbose = $(tuner.verbose)")
