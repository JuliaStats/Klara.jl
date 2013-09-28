###########################################################################
#
#  Robust adaptive Metropolis (RAM)
#
#  Reference: Vihola M. Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate. Statistics and Computing,
#  2012, 22 (5), pp 997-1008.
#
#  Parameters:
#    -scale: for scaling the jumps
#    -rate: target acceptance rate
#
###########################################################################

export RAM, RAMTuner

println("Loading RAM(scale, rate) sampler")

###########################################################################
#                  RAM type
###########################################################################

immutable RAM <: MCMCSampler
  scale::Float64
  rate::Float64

  function RAM(x::Float64, r::Float64)
    assert(x>0, "scale should be > 0")
    assert(r > 0. && r < 1., "target acceptance rate ($r) should be between 0 and 1")
    new(x, r)
  end
end
RAM() = RAM(1., 0.234)
RAM(x::Float64) = RAM(x, 0.234)
RAM(;scale::Float64=1.0, rate::Float64=0.234) = RAM(scale, rate)

###########################################################################
#                  RAM task
###########################################################################

# RAM sampling
function SamplerTask(model::MCMCModel, sampler::RAM, runner::MCMCRunner)
	local pars, proposedPars
	local logTarget, proposedLogTarget
  local proposed
  local scale

	# hook inside Task to allow remote resetting
	task_local_storage(:reset, (resetPars::Vector{Float64}) -> (pars = copy(resetPars); logTarget = model.eval(pars))) 

	# initialization
	scale = model.scale .* sampler.scale  # rescale model scale by sampler scale
	pars = copy(model.init)
	logTarget = model.eval(pars)

	S = Float64[ i==j ? scale[i] : 0. for i in 1:model.size, j in 1:model.size]
		
	# main loop
	for i in 1:Inf
		rvec = randn(model.size)
		proposedPars = pars + S * rvec
		proposedLogTarget = model.eval(proposedPars) 

	  ratio = proposedLogTarget-logTarget
		if ratio > 0 || (ratio > log(rand()))
			ms = MCMCSample(proposedPars, proposedLogTarget, pars, logTarget, {"accept" => true, "scale" => trace(S)})
      produce(ms)
      pars, logTarget = proposedPars, proposedLogTarget
    else
			ms = MCMCSample(pars, logTarget, pars, logTarget, {"accept" => false, "scale" => trace(S)})
      produce(ms)
    end

    # scale tuning
		eta = min(1, model.size * i^(-2/3))  # decreasing influence with i

		SS = (rvec * rvec') / dot(rvec, rvec) * eta * (min(1, exp(ratio)) - sampler.rate)
		SS = S * (eye(model.size) + SS) * S'
		S = chol(SS)'  #'
  end
end
