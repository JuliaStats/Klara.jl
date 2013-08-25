###########################################################################
#
#  Random-Walk Metropolis (RWM)
#
#  Parameters:
#    -scale: for scaling the jumps
#    -tuner: for tuning the scale parameter
#
###########################################################################

export RWM

println("Loading RMW(scale, tuner) sampler")

#### RWM specific 'tuners'
abstract RWMTuner <: MCMCTuner

# TODO 1: define scale tuner
immutable RWMEmpiricalTuner <: RWMTuner
  rater::AcceptanceRater

  function RWMEmpiricalTuner(rater::AcceptanceRater)  	
  end
end	

####  RWM sampler type  ####
immutable RWM <: MCMCSampler
  scale::Float64
  tuner::Union(nothing, RWMTuner)

  function RWM(x::Float64, t::Union(nothing, RWMTuner))
    assert(x>0, "scale should be > 0")
    new(x, t)
  end
end
RWM() = RWM(1., nothing)
RWM(x::Float64) = RWM(x, nothing)
RWM(t::Union(nothing, RWMTuner)) = RWM(1., t)

# Sampling task launcher
spinTask(model::MCMCModel, s::RWM) = MCMCTask(Task(() -> RWMTask(model, s.scale, s.tuner)), model)

# RWM sampling
function RWMTask(model::MCMCModel, scale::Float64, tuner::Union(nothing, RWMTuner))
	local pars, proposedPars
	local logTarget, proposedLogTarget
    local proposed, accepted

	#  Task reset function
	function reset(resetPars::Vector{Float64})
		proposedPars = copy(resetPars)
		proposedLogTarget = model.eval(proposedPars)
	end
	# hook inside Task to allow remote resetting
	task_local_storage(:reset, reset) 

	# initialization
	proposedPars = copy(model.init)
	proposedLogTarget = model.eval(proposedPars)
	assert(isfinite(proposedLogTarget), "Initial values out of model support, try other values")

	while true
		pars = copy(proposedPars)
		# TODO 2: if tuner != nothing; tune the scale; end
		proposedPars += randn(model.size) * model.scale

 		logTarget, proposedLogTarget = proposedLogTarget, model.eval(proposedPars) 

		if rand() > exp(proposedLogTarget - logTarget) # roll back if rejected
			proposedLogTarget, proposedPars = logTarget, pars
		end

		produce(MCMCSample(proposedPars, proposedLogTarget, pars, logTarget))
	end
end
