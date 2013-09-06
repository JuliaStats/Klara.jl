###########################################################################
#
#  Random-Walk Metropolis (RWM)
#
#  Parameters:
#    -scale: for scaling the jumps
#    -tuner: for tuning the scale parameter
#
###########################################################################

export RWM, RAM

println("Loading RMW(scale, tuner) sampler")

###########################################################################
#                  RWM specific 'tuners'
###########################################################################
abstract RWMTuner <: MCMCTuner

# TODO 1: define scale tuner
immutable RWMEmpiricalTuner <: RWMTuner
  rate::AcceptanceRate

  function RWMEmpiricalTuner(rate::AcceptanceRate)
    new(rate) 	
  end
end	

# Robust Adaptative Scaling
#  ref: ROBUST ADAPTIVE METROPOLIS ALGORITHM WITH COERCED ACCEPTANCE RATE - Matti Vihola
# TODO   : add an explicit adaptation period
# TODO 2 : add an adaptation Rate (currently every step)
immutable RAM <: RWMTuner
  targetacceptrate::Float64

  function RAM(r::Float64)
  	assert(r > 0. && r < 1., "target acceptance rate ($r) should be between 0 and 1")
    new(r) 	
  end
end	
RAM() = RAM(0.234)  # default target acceptance of 23.4%


###########################################################################
#                  RWM type
###########################################################################

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
RWM(t::RWMTuner) = RWM(1., t)
RWM(;scale::Float64=1.0, tuner::Union(Nothing, RWMTuner)=nothing) = RWM(scale, tuner)

###########################################################################
#                  RWM task
###########################################################################

# RWM sampling
function SamplerTask(model::MCMCModel, sampler::RWM)
	local pars, proposedPars
	local logTarget, proposedLogTarget
    local proposed
    local scale

	# hook inside Task to allow remote resetting
	task_local_storage(:reset,
					   (resetPars::Vector{Float64}) -> (pars = copy(resetPars); 
														logTarget = model.eval(pars)) ) 

	# initialization
	scale = model.scale .* sampler.scale  # rescale model scale by sampler scale
	pars = copy(model.init)
	logTarget = model.eval(pars)

	# two different loops depending on tuner
	#  TODO : when tuning changes so much, may be we should just make a new MCMCSampler altogether ??
	if isa(sampler.tuner, RAM)  # RAM tuner
		# start with diagonal matrix build with scale
		S = Float64[ i==j ? scale[i] : 0. for i in 1:model.size, j in 1:model.size]
		
		# main loop
		for i in 1:Inf
			rvec = randn(model.size)
			proposedPars = pars + S * rvec
	 		proposedLogTarget = model.eval(proposedPars) 

	        ratio = proposedLogTarget-logTarget
			if ratio > 0 || (ratio > log(rand()))
				ms = MCMCSample(proposedPars, proposedLogTarget, 
								pars, logTarget,
								{"accept" => true, "scale" => trace(S)} )
	        	produce(ms)
	         	pars, logTarget = proposedPars, proposedLogTarget
	        else
				ms = MCMCSample(pars, logTarget, 
								pars, logTarget,
								{"accept" => false, "scale" => trace(S)} )
	     		produce(ms)
	    	end

	    	# scale tuning
			eta = min(1, model.size * i^(-2/3))  # decreasing influence with i

			SS = (rvec * rvec') / dot(rvec, rvec) * 
				eta * (min(1, exp(ratio)) - sampler.tuner.targetacceptrate)
			SS = S * (eye(model.size) + SS) * S'
			S = chol(SS)'  #'
		end

	else   # no tuner case
		# main loop
		for i in 1:Inf
			proposedPars = pars + randn(model.size) .* scale
	 		proposedLogTarget = model.eval(proposedPars) 

	        ratio = proposedLogTarget-logTarget
			if ratio > 0 || (ratio > log(rand()))
				ms = MCMCSample(proposedPars, proposedLogTarget, 
								pars, logTarget,
								{"accept" => true} )
	        	produce(ms)
	         	pars, logTarget = proposedPars, proposedLogTarget
	        else
				ms = MCMCSample(pars, logTarget, 
								pars, logTarget,
								{"accept" => false} )
	     		produce(ms)
	    	end
		end

	end

end
