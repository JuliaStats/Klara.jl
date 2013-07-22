###########################################################################
#  Random-Walk Metropolis sampler
#
#     takes a scalar as parameter to give a scale to jumps
#
###########################################################################

export RWM

println("Loading RMW(scale) sampler")

# RWM sampler type
immutable RWM <: MCMCSampler
  scale::Float64

  function RWM(x::Real)
    assert(x>0, "scale should be > 0")
    new(x)
  end
end
RWM() = RWM(1.)


# sampling task launcher
spinTask(model::MCMCModel, s::RWM) = 
	MCMCTask( Task(() -> RWMTask(model, s.scale)), model)

# RWM sampling
function RWMTask(model::MCMCModel, scale::Float64)
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
		beta += randn(model.size) * scale

 		oldll, ll = ll, model.eval(beta) 

		if rand() > exp(ll - oldll) # roll back if rejected
			ll, beta = oldll, oldbeta
		end

		produce(MCMCSample(beta, ll, oldbeta, oldll))
	end
end

