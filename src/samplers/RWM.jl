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

println("Loading RWM(scale, tuner) sampler")

###########################################################################
#                  RWM specific 'tuners'
###########################################################################
abstract RWMTuner <: MCMCTuner

###########################################################################
#                  RWM type
###########################################################################

immutable RWM <: MCMCSampler
  scale::Float64
  tuner::Union(Nothing, RWMTuner)

  function RWM(x::Float64, t::Union(Nothing, RWMTuner))
    @assert x>0 "scale should be > 0"
    new(x, t)
  end
end
RWM() = RWM(1., nothing)
RWM(x::Float64) = RWM(x, nothing)
RWM(t::RWMTuner) = RWM(1., t)
RWM(;scale::Float64=1.0, tuner::Union(Nothing, RWMTuner)=nothing) = RWM(scale, tuner)

###########################################################################
#                  RWM task
###########################################################################

# RWM sampling
function SamplerTask(model::MCMCModel, sampler::RWM, runner::MCMCRunner)
	local pars, proposedPars
	local logTarget, proposedLogTarget
  local scale

	# hook inside Task to allow remote resetting
	task_local_storage(:reset, (resetPars::Vector{Float64}) -> (pars = copy(resetPars); logTarget = model.eval(pars))) 

	# initialization
	scale = model.scale .* sampler.scale  # rescale model scale by sampler scale
	pars = copy(model.init)
	logTarget = model.eval(pars)
  @assert isfinite(logTarget) "Initial values out of model support, try other values"
  
	# main loop
	while true
		proposedPars = pars + randn(model.size) .* scale
		proposedLogTarget = model.eval(proposedPars) 

	  ratio = proposedLogTarget-logTarget
		if ratio > 0 || (ratio > log(rand()))
			ms = MCMCSample(proposedPars, proposedLogTarget, pars, logTarget, {"accept" => true})
	    produce(ms)
	    pars, logTarget = proposedPars, proposedLogTarget
	  else
			ms = MCMCSample(pars, logTarget, pars, logTarget, {"accept" => false})
      produce(ms)
    end
	end
end
