###########################################################################
#
#  Independent Metropolis-Hastings (IMH)
#
#  Parameters:
#    -logCandidate: candidate density
#    -randCandidate: function for sampling from candidate density
#
###########################################################################

export IMH

println("Loading IMH(logCandidate, randCandidate!) sampler")

###########################################################################
#                  IMH type
###########################################################################

immutable IMH <: MCMCSampler
  logCandidate::Function
  randCandidate!::Function
end

IMH(proposal::ContinuousMultivariateDistribution) =
  IMH(pars::Vector{Float64} -> logpdf(proposal, pars), sample::Vector{Float64} -> rand!(proposal, sample))

###########################################################################
#                  IMH task
###########################################################################

# IMH sampling
function SamplerTask(model::MCMCModel, sampler::IMH, runner::MCMCRunner)
	local pars, proposedPars
	local logTarget, logCandidate, proposedLogTarget, proposedLogCandidate

	# hook inside Task to allow remote resetting
	task_local_storage(:reset, (resetPars::Vector{Float64}) -> (pars = copy(resetPars); logTarget = model.eval(pars))) 

	# initialization
	pars, proposedPars = copy(model.init), zeros(model.size)
	logTarget, logCandidate = model.eval(pars), sampler.logCandidate(pars)
  @assert isfinite(logTarget) "Initial values out of model support, try other values"
  @assert isfinite(logCandidate) "Initial values out of candidate density support, try other values"

	# main loop
	while true
		sampler.randCandidate!(proposedPars)
		proposedLogTarget, proposedLogCandidate = model.eval(proposedPars), sampler.logCandidate(proposedPars)

	  ratio = proposedLogTarget-logTarget-proposedLogCandidate+logCandidate
		if ratio > 0 || (ratio > log(rand()))
			ms = MCMCSample(proposedPars, proposedLogTarget, pars, logTarget, {"accept" => true})
	    produce(ms)
	    pars, logTarget, logCandidate = proposedPars, proposedLogTarget, proposedLogCandidate
	  else
			ms = MCMCSample(pars, logTarget, pars, logTarget, {"accept" => false})
      produce(ms)
    end
	end
end
