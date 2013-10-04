#################################################################
#
#    Common defs for samplers
#
#################################################################

export EmpMCTuner

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

######### Empirical tuner common for various samplers ##
immutable EmpiricalMCMCTuner <: MCMCTuner
  adaptStep::Int
  maxStep::Int
  targetPath::Float64
  targetRate::Float64
  verbose::Bool

  function EmpiricalMCMCTuner(adaptStep::Int, maxStep::Int, targetPath::Float64, targetRate::Float64, verbose::Bool)
    assert(adaptStep > 0, "Adaptation step size ($adaptStep) should be > 0")
    assert(maxStep > 0, "Adaptation step size ($maxStep) should be > 0")    
    assert(0 < targetRate < 1, "Target acceptance rate ($targetRate) should be between 0 and 1")
    new(adaptStep, maxStep, targetPath, targetRate, verbose)
  end
end

typealias EmpMCTuner EmpiricalMCMCTuner

EmpMCTuner(targetRate::Float64; adaptStep::Int=100, maxStep::Int=200, targetPath::Float64=1., verbose::Bool=false) =
  EmpMCTuner(adaptStep, maxStep, targetPath, targetRate, verbose)

# Sampling task launcher
spinTask(m::MCMCModel, s::MCMCSampler, r::MCMCRunner) = MCMCTask(Task(() -> SamplerTask(m, s, r)), m, s, r)
