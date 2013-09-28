###########################################################################
#  Position-Dependent Metropolis adjusted Langevin algorithm (PMALA)
#
#  Reference: Xifara T, Sherlock C, Livingstone S, Byrne S, Girolami M. Langevin Diffusions and the Metropolis-Adjusted
# Langevin Algorithm. arXiv, 2013
#
#  Parameters :
#    - driftStep : drift step size (for scaling the jumps)
#    - tuner: for tuning the drift step size
#
###########################################################################

export PMALA

println("Loading PMALA(driftStep, tuner) sampler")

###########################################################################
# PMALA specific tuners
###########################################################################
abstract PMALATuner <: MCMCTuner

###########################################################################
# PMALA type
###########################################################################
immutable PMALA <: MCMCSampler
  driftStep::Float64
  tuner::Union(Nothing, PMALATuner)
  
  function PMALA(s::Real, t::Union(Nothing, PMALATuner))
    assert(s>0, "PMALA drift step should be > 0")
    new(s, t)
  end
end

PMALA(s::Float64=1.0) = PMALA(s, nothing)
PMALA(s::MALATuner) = PMALA(1.0, t)
PMALA(;scale::Float64=1.0, tuner::Union(Nothing, PMALATuner)=nothing) = PMALA(scale, tuner)

###########################################################################
# PMALA sampler
###########################################################################
function SamplerTask(model::MCMCModel, sampler::PMALA, runner::MCMCRunner)
  # Define local variables
  local pars, proposedPars, parsMean
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local H, proposedH
  local G, invG, cholOfInvG, dG, invGxdG, proposedG, proposedInvG
  local firstTerm, secondTerm, proposedFirstTerm, proposedSecondTerm
  local probNewGivenOld, probOldGivenNew

  # Allocate memory for some of the local variables
  invGxdG = Array(Float64, model.size, model.size, model.size)
  secondTerm = Array(Float64, model.size, model.size)
  proposedSecondTerm = Array(Float64, model.size, model.size)

  # Task reset function
  function reset(resetPars::Vector{Float64})
    pars = copy(resetPars)
    logTarget, grad, G, dG = model.evalalldt(pars)
  end
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 

  # Initialization
  pars = copy(model.init)
  logTarget, grad, G, dG = model.evalalldt(pars)
  invG = inv(G)
  firstTerm = invG*grad
  for i = 1:model.size    
    invGxdG[:, :, i] = invG*dG[:, :, i]
    secondTerm[:, i] = invGxdG[:, :, i]*invG[:, i]
  end
  
  while true
    # Calculate the drift term
    parsMean = pars+(sampler.driftStep/2)*(firstTerm-sum(secondTerm, 2)[:])

    # Calculate proposed parameters
    cholOfInvG = chol(sampler.driftStep*invG)
    proposedPars = parsMean+cholOfInvG'*randn(model.size)

    # Update model based on the proposed parameters
    proposedLogTarget, proposedGrad, proposedG, dG = model.evalalldt(proposedPars)

    probNewGivenOld = (-sum(log(diag(cholOfInvG)))
      -(0.5*(parsMean-proposedPars)'*(G/sampler.driftStep)*(parsMean-proposedPars))[1])

    proposedInvG = inv(proposedG)
    proposedFirstTerm = proposedInvG*proposedGrad   
    for j = 1:model.size               
      invGxdG[:, :, j] = proposedInvG*dG[:, :, j]
      proposedSecondTerm[:, j] = invGxdG[:, :, j]*proposedInvG[:, j]
    end

    # Calculate the drift term
    parsMean = proposedPars+(sampler.driftStep/2)*(proposedFirstTerm-sum(proposedSecondTerm, 2)[:])

    probOldGivenNew = (-sum(log(diag(chol(sampler.driftStep*eye(model.size)*proposedInvG))))
      -(0.5*(parsMean-pars)'*(proposedG/sampler.driftStep)*(parsMean-pars))[1])
 
    ratio = proposedLogTarget+probOldGivenNew-logTarget-probNewGivenOld

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad, {"accept" => true}))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)
      G, invG = copy(proposedG), copy(proposedInvG)
      firstTerm, secondTerm = copy(proposedFirstTerm), copy(proposedSecondTerm)
    else
      produce(MCMCSample(pars, logTarget, grad, pars, logTarget, grad, {"accept" => false}))
    end
  end
end
