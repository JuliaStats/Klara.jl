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
  rate::AcceptanceRate

  function RWMEmpiricalTuner(rate::AcceptanceRate)
    new(rate) 	
  end
end	

####  RWM sampler type  ####
immutable RWM <: MCMCSampler
  scale::Float64
  tuner::Union(Nothing, RWMTuner)

  function RWM(x::Float64, t::Union(Nothing, RWMTuner))
    assert(x>0, "scale should be > 0")
    new(x, t)
  end
end
RWM() = RWM(1., nothing)
RWM(x::Float64) = RWM(x, nothing)
RWM(t::Union(Nothing, RWMTuner)) = RWM(1., t)

# Sampling task launcher
spinTask(model::MCMCModel, s::RWM) = MCMCTask(Task(() -> RWMTask(model, s.scale, s.tuner)), model)

# RWM sampling
function RWMTask(model::MCMCModel, scale::Float64, tuner::Union(Nothing, RWMTuner))
	local pars, proposedPars
	local logTarget, proposedLogTarget
    local proposed, accepted

	#  Task reset function
	function reset(resetPars::Vector{Float64})
		pars = copy(resetPars)
		logTarget = model.eval(pars)
	end
	# hook inside Task to allow remote resetting
	task_local_storage(:reset, reset) 

	# initialization
	pars = copy(model.init)
	logTarget = model.eval(pars)
	assert(isfinite(logTarget), "Initial values out of model support, try other values")

	while true
		proposedPars = copy(pars)
		# TODO 2: if tuner != nothing; tune the scale; end
		proposedPars += randn(model.size) * scale
 		proposedLogTarget = model.eval(proposedPars) 

        ratio = proposedLogTarget-logTarget
		if ratio > 0 || (ratio > log(rand()))
        	produce(MCMCSample(proposedPars, proposedLogTarget, pars, logTarget))
         	pars, logTarget = proposedPars, proposedLogTarget
        else
     		produce(MCMCSample(pars, logTarget, pars, logTarget))
    	end
	end
end
