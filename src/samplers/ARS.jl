###########################################################################
#
#  Acceptance Rejection Sampling (ARS)
#
#  Parameters:
#    -g0: unscaled candidate
#    -scale: for scaling the jumps
#    -tuner: for tuning the scale parameter
#
###########################################################################

export ARS

println("Loading ARS(g0, scale, tuner) sampler")

###########################################################################
#                  ARS specific 'tuners'
###########################################################################
abstract ARSTuner <: MCMCTuner

###########################################################################
#                  ARS type
###########################################################################

immutable ARS <: MCMCSampler
  g0::Function
  lM::Float64
  scale::Float64
  tuner::Union(Nothing, ARSTuner)

  function ARS(g0::Function, lM::Float64, x::Float64, t::Union(Nothing, ARSTuner))
    @assert typeof(g0) == Function "Unscaled candidate g0 should be a function"
    @assert x>0 "scale should be > 0"
    new(g0, lM, x, t)
  end
end
ARS(g0::Function) = ARS(g0, 1., 1., nothing)
ARS(g0::Function, lM::Float64) = ARS(g0, lM, 1.0, nothing)
ARS(g0::Function, t::ARSTuner) = ARS(g0, 1., 1., t)
ARS(g0::Function; lM::Float64 = 1.0, scale::Float64=1.0, tuner::Union(Nothing, ARSTuner)=nothing) = ARS(g0, M, scale, tuner)

###########################################################################
#                  ARS task
###########################################################################

# ARS sampling
function SamplerTask(model::MCMCModel, sampler::ARS, runner::MCMCRunner)
	local pars, proposedPars
	local logTarget, proposedLogTarget, proposedLogCandidate
  local scale, M, weight, mu, count

	# hook inside Task to allow remote resetting
	task_local_storage(:reset, (resetPars::Vector{Float64}) -> (pars = copy(resetPars); logTarget = model.eval(pars))) 

	# initialization
	scale = model.scale .* sampler.scale  # rescale model scale by sampler scale
  println(sampler.lM)
	pars = copy(model.init)
	logTarget = model.eval(pars)
  @assert isfinite(logTarget) "Initial values out of model support, try other values"
  
	# main loop
  count = 0
	while true
    if count > 1
		  proposedPars = pars + randn(model.size) .* scale
      mu = log(rand())
    else
      if count == 0
        proposedPars = [2.6]
        mu = log(0.415198)
      else
        proposedPars = [-0.94]
        mu = log(0.577230)
      end
      count += 1
    end  
		proposedLogTarget = model.eval(proposedPars) 
		proposedLogCandidate = sampler.g0(proposedPars) 
	  weight = proposedLogTarget .- sampler.lM .- proposedLogCandidate
    println([proposedPars exp(sampler.g0(proposedPars[1])) exp(weight) exp(mu)])
		if weight > mu
			ms = MCMCSample(proposedPars, proposedLogTarget, pars, logTarget, {"accept" => true})
	    produce(ms)
	    pars, logTarget = copy(proposedPars), copy(proposedLogTarget)
	  else
			ms = MCMCSample(pars, logTarget, pars, logTarget, {"accept" => false})
      produce(ms)
    end
	end
end
