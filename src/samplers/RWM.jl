###########################################################################
#
#  Random-Walk Metropolis sampler
#
#     takes a scalar as parameter to give a scale to jumps
#
###########################################################################

export RWM

println("Loading RMW(scale) sampler")

#### RWM specific 'tuners'
immutable DummyTuner  <: MCMCTuner
end	

immutable DummyTuner2  <: MCMCTuner
end	


####  RWM sampler type  ####
immutable RWM <: MCMCSampler
  tuner::MCMCTuner
end
RWM() = RWM(DummyTuner())

# sampling task launcher
spinTask(model::MCMCModel, s::RWM) = 
	MCMCTask( Task(() -> RWMTask(model, s)) , model )

# RWM sampling
function RWMTask(model::MCMCModel, sampler::MCMCSampler)
	local beta, ll, oldbeta, oldll

	#  Task reset function
	function reset(newbeta::Vector{Float64})
		beta = copy(newbeta)
		ll = model.eval(beta)
	end
	# hook inside Task to allow remote resetting
	task_local_storage(:reset, reset) 

	# initialization
	beta = copy(model.init)
	ll = model.eval(beta)
	assert(ll != -Inf, "Initial values out of model support, try other values")

	while true
		oldbeta = copy(beta)
		beta += randn(model.size) * model.scale

 		oldll, ll = ll, model.eval(beta) 

		if rand() > exp(ll - oldll) # roll back if rejected
			ll, beta = oldll, oldbeta
		end

		produce(MCMCSample(beta, ll, oldbeta, oldll))
	end
end

