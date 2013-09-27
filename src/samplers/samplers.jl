#################################################################
#
#    Common defs for samplers
#
#################################################################

abstract MCMCSampler
abstract MCMCTuner

######### sample type returned by samplers  ############
immutable MCMCSample
	ppars::Vector{Float64}                  # proposed parameter vector
	plogtarget::Float64                     # log target	at proposed parameter vector
  pgrads::Union(Vector{Float64}, Nothing) # grad of log target at proposed parameter vector
	pars::Vector{Float64}                   # parameter vector before proposal
	logtarget::Float64                      # log target before proposal
  grads::Union(Vector{Float64}, Nothing)  # grad of log target before proposal
	diagnostics::Dict{Any,Any}              # sampler-dependant diagnostic variables
end

MCMCSample(ppars::Vector{Float64}, plogtarget::Float64,
  pars::Vector{Float64}, logtarget::Float64, diagnostics::Dict{Any,Any}) =
  MCMCSample(ppars, plogtarget, nothing, pars, logtarget, nothing, diagnostics)

MCMCSample(ppars::Vector{Float64}, plogtarget::Float64, pars::Vector{Float64}, logtarget::Float64) =
  MCMCSample(ppars, plogtarget, pars, logtarget, Dict())

MCMCSample(ppars::Vector{Float64}, plogtarget::Float64, pgrads::Union(Vector{Float64}, Nothing),
  pars::Vector{Float64}, logtarget::Float64, grads::Union(Vector{Float64}, Nothing)) =
  MCMCSample(ppars, plogtarget, pgrads, pars, logtarget, grads, Dict())

# Sampling task launcher
spinTask(m::MCMCModel, s::MCMCSampler) = MCMCTask(Task(() -> SamplerTask(m, s)), m)
