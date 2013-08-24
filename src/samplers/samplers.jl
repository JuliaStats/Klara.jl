#################################################################
#
#    Common defs for samplers
#
#################################################################

abstract MCMCSampler
abstract MCMCTuner

######### sample type returned by samplers  ############
immutable MCMCSample
	ppars::Vector{Float64} # proposed parameter vector
	plogtarget::Float64    # proposed log target	
	pars::Vector{Float64}  # parameter vector before proposal
	logtarget::Float64     # log target before proposal
end
# TODO : add diagnostic field to MCMCSample
