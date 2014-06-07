###########################################################################
#
#  Acceptance Rejection Sampling (ARS)
#
#  Parameters:
#    -logCandidate: unscaled candidate
#    -logCandidateScalingFactor: scale factor to ensure the scaled-up
#        logCandidate covers target
#    -scale: for scaling the jumps
#    -tuner: for tuning the scale parameter
#
###########################################################################

###########################################################################
#
# Based on Bolstad: Computational Bayesian Statistics, section 2.1
#
###########################################################################

export ARS

println("Loading ARS(logCandidate, logCandidateScalingFactor, scale, tuner) sampler")

###########################################################################
#                  ARS specific 'tuners'
###########################################################################
abstract ARSTuner <: MCMCTuner

###########################################################################
#                  ARS type
###########################################################################

immutable ARS <: MCMCSampler
  logCandidate::Function
  logCandidateScalingFactor::Float64
  scale::Float64
  tuner::Union(Nothing, ARSTuner)

  function ARS(logCandidate::Function, logCandidateScalingFactor::Float64, x::Float64, t::Union(Nothing, ARSTuner))
    @assert typeof(logCandidate) == Function "Unscaled candidate logCandidate should be a function"
    @assert x>0 "scale should be > 0"
    new(logCandidate, logCandidateScalingFactor, x, t)
  end
end
ARS(logCandidate::Function) = ARS(logCandidate, 1., 1., nothing)
ARS(logCandidate::Function, logCandidateScalingFactor::Float64) = ARS(logCandidate, logCandidateScalingFactor, 1.0, nothing)
ARS(logCandidate::Function, t::ARSTuner) = ARS(logCandidate, 1., 1., t)
ARS(logCandidate::Function; logCandidateScalingFactor::Float64 = 1.0, scale::Float64=1.0, tuner::Union(Nothing, ARSTuner)=nothing) = ARS(logCandidate, logCandidateScalingFactor, scale, tuner)

###########################################################################
#                  ARS task
###########################################################################

# ARS sampling
function SamplerTask(model::MCMCModel, sampler::ARS, runner::MCMCRunner)
	local pars, proposedPars
	local logTarget, proposedLogTarget, proposedLogCandidate
  local scale, weight, mu

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
    mu = log(rand())
    proposedLogTarget = model.eval(proposedPars) 
		proposedLogCandidate = sampler.logCandidate(proposedPars)
	  weight = proposedLogTarget .- sampler.logCandidateScalingFactor .- proposedLogCandidate
		if weight > mu
			ms = MCMCSample(proposedPars, proposedLogTarget, pars, logTarget, 
        {"accept" => true})
	    produce(ms)
	    pars, logTarget = copy(proposedPars), copy(proposedLogTarget)
	  else
			ms = MCMCSample(pars, logTarget, pars, logTarget, 
        {"accept" => false})
      produce(ms)
    end
	end
end
