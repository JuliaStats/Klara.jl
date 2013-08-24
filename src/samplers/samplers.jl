#################################################################
#
#    Common defs for samplers
#
#################################################################

abstract MCMCSampler
abstract MCMCTuner

######### sample type returned by samplers  ############
immutable MCMCSample
	beta::Vector{Float64}     # newly drawn parameter vector
	ll::Float64               # log likelihood	
	oldbeta::Vector{Float64}  # previous vector
	oldll::Float64            # previous log likelihood	
end
# TODO 1 : add diagnostic field to MCMCSample
# TODO 2 : change beta to pars, oldbeta to oldpars (and perhaps ll to logtarget, oldll to oldlogtarget.
# The reason target seems a preferable name over lik is because the target can be either the posterior
# in a Bayesian context or just a distribution (likelihood) if we don't want to use a proper prior in our model)
