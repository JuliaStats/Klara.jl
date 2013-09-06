###########################################################################
#  Simplified manifold Metropolis adjusted Langevin algorithm (SMMALA)
#
#  Parameters :
#    - driftStep : drift step size (for scaling the jumps)
#    - tuner: for tuning the drift step size
#
###########################################################################

export SMMALA

println("Loading SMMALA(driftStep, tuner) sampler")

###########################################################################
# SMMALA specific tuners
###########################################################################
abstract SMMALATuner <: MCMCTuner

###########################################################################
# SMMALA type
###########################################################################
immutable SMMALA <: MCMCSampler
  driftStep::Float64
  tuner::Union(Nothing, SMMALATuner)
  
  function SMMALA(s::Real, t::Union(Nothing, SMMALATuner))
    assert(s>0, "SMMALA drift step should be > 0")
    new(s, t)
  end
end

SMMALA(s::Float64=1.0) = SMMALA(s, nothing)
SMMALA(s::MALATuner) = SMMALA(1.0, t)
SMMALA(;scale::Float64=1.0, tuner::Union(Nothing, SMMALATuner)=nothing) = SMMALA(scale, tuner)

###########################################################################
# SMMALA sampler
###########################################################################
function SamplerTask(model::MCMCModel, sampler::SMMALA)
  # Define local variables
  local pars, proposedPars, parsMean
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local H, proposedH
  local G, invG, cholOfInvG, proposedG, proposedInvG
  local firstTerm, proposedFirstTerm
  local probNewGivenOld, probOldGivenNew

   # Task reset function
  function reset(resetPars::Vector{Float64})
    pars = copy(resetPars)
    logTarget, grad, G = model.evalallt(pars)
  end
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 

  # Initialization
  pars = copy(model.init)
  logTarget, grad, G = model.evalallt(pars)
  invG = inv(G)
  firstTerm = invG*grad
  
  while true
    # Calculate the drift term
    parsMean = pars+(sampler.driftStep/2)*firstTerm

    # Calculate proposed parameters
    cholOfInvG = chol(sampler.driftStep*invG)
    proposedPars = parsMean+cholOfInvG'*randn(model.size)

    # Update model based on the proposed parameters
    proposedLogTarget, proposedGrad, proposedG = model.evalallt(proposedPars)

    probNewGivenOld = (-sum(log(diag(cholOfInvG)))
      -(0.5*(parsMean-proposedPars)'*(G/sampler.driftStep)*(parsMean-proposedPars))[1])

    proposedInvG = inv(proposedG)
    proposedFirstTerm = proposedInvG*proposedGrad

    # Calculate the drift term
    parsMean = proposedPars+(sampler.driftStep/2)*proposedFirstTerm

    probOldGivenNew = (-sum(log(diag(chol(sampler.driftStep*eye(model.size)*proposedInvG))))
      -(0.5*(parsMean-pars)'*(proposedG/sampler.driftStep)*(parsMean-pars))[1])
 
    ratio = proposedLogTarget+probOldGivenNew-logTarget-probNewGivenOld

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)
      G, invG, firstTerm = copy(proposedG), copy(proposedInvG), copy(proposedFirstTerm)
    else
      produce(MCMCSample(pars, logTarget, grad, pars, logTarget, grad))
    end
  end
end
