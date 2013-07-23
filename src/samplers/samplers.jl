#################################################################
#
#    Common defs for samplers
#
#################################################################

abstract MCMCSampler

######### sample type returned by samplers  ############
immutable MCMCSample
	beta::Vector{Float64}     # newly drawn parameter vector
	ll::Float64               # log likelihood	
	oldbeta::Vector{Float64}  # previous vector
	oldll::Float64            # previous log likelihood	
end

