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
spinTask(model::MCMCModel, s::RWM) = MCMCTask( Task(() -> RWMTask(model, s.scale)), model)

# RWM sampling
function RWMTask(model::MCMCModel, scale::Float64)
	local beta = copy(model.init)
	local ll = model.eval(beta)
	local oldbeta, oldll, jump

	assert(ll != -Inf, "Initial values out of model support, try other values")

	task_local_storage(:reset, (x::Vector{Float64}) -> beta=x)  # hook inside Task to allow remote resetting

	while true
		jump = randn(model.size) * scale
		oldbeta = copy(beta)
		beta += jump 

 		oldll, ll = ll, model.eval(beta) 

		if rand() > exp(ll - oldll) # roll back if rejected
			ll, beta = oldll, oldbeta
		end

		produce(beta)
	end
end

