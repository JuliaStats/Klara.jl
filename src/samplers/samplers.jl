#################################################################
#
#    Common defs for samplers
#
#################################################################

abstract MCMCSampler
abstract MCMCTuner

immutable AcceptanceRate
  step::Integer

  function AcceptanceRate(s::Integer)
    assert(s>0, "Acceptance ratio's monitor step should be > 0")
    new(s)
  end
end

######### sample type returned by samplers  ############
immutable MCMCSample
	ppars::Vector{Float64}     # proposed parameter vector
	plogtarget::Float64        # proposed log target	
	pars::Vector{Float64}      # parameter vector before proposal
	logtarget::Float64         # log target before proposal
	diagnostics::Dict{Any,Any} # sampler-dependant diagnostic variables
end

MCMCSample(ppars::Vector{Float64}, plogtarget::Float64, pars::Vector{Float64}, logtarget::Float64) =
  MCMCSample(ppars, plogtarget, pars, logtarget, Dict())


# Sampling task launcher
spinTask(m::MCMCModel, s::MCMCSampler) = MCMCTask(Task(() -> SamplerTask(m, s)), m)

