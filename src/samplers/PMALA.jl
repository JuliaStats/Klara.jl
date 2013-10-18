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
  tuner::Union(Nothing, MCMCTuner)
  
  function PMALA(s::Real, t::Union(Nothing, MCMCTuner))
    @assert s>0 "PMALA drift step should be > 0"
    new(s, t)
  end
end

PMALA(s::Float64=1.0) = PMALA(s, nothing)
PMALA(s::MALATuner) = PMALA(1.0, t)
PMALA(;scale::Float64=1.0, tuner::Union(Nothing, MCMCTuner)=nothing) = PMALA(scale, tuner)

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
  local driftStep

  @assert hasgradient(model) "PMALA sampler requires model with gradient function"
  @assert hastensor(model) "PMALA sampler requires model with tensor function"
  @assert hasdtensor(model) "PMALA sampler requires model with function of tensor derivatives"

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
  @assert isfinite(logTarget) "Initial values out of model support, try other values"

  invG = inv(G)
  firstTerm = invG*grad
  for i = 1:model.size    
    invGxdG[:, :, i] = invG*dG[:, :, i]
    secondTerm[:, i] = invGxdG[:, :, i]*invG[:, i]
  end
  
  if isa(sampler.tuner, EmpMCTuner); tune = EmpiricalMALATune(sampler.driftStep, 0, 0); end

  i = 1
  while true
    if isa(sampler.tuner, EmpMCTuner)
      tune.proposed += 1
      driftStep = tune.driftStep
    else
      driftStep = sampler.driftStep
    end

    # Calculate the drift term
    parsMean = pars+(driftStep/2)*(firstTerm-sum(secondTerm, 2)[:])

    # Calculate proposed parameters
    cholOfInvG = chol(driftStep*invG)
    proposedPars = parsMean+cholOfInvG'*randn(model.size)

    # Update model based on the proposed parameters
    proposedLogTarget, proposedGrad, proposedG, dG = model.evalalldt(proposedPars)

    probNewGivenOld = (-sum(log(diag(cholOfInvG)))
      -(0.5*(parsMean-proposedPars)'*(G/driftStep)*(parsMean-proposedPars))[1])

    proposedInvG = inv(proposedG)
    proposedFirstTerm = proposedInvG*proposedGrad   
    for j = 1:model.size               
      invGxdG[:, :, j] = proposedInvG*dG[:, :, j]
      proposedSecondTerm[:, j] = invGxdG[:, :, j]*proposedInvG[:, j]
    end

    # Calculate the drift term
    parsMean = proposedPars+(driftStep/2)*(proposedFirstTerm-sum(proposedSecondTerm, 2)[:])

    probOldGivenNew = (-sum(log(diag(chol(driftStep*eye(model.size)*proposedInvG))))
      -(0.5*(parsMean-pars)'*(proposedG/driftStep)*(parsMean-pars))[1])
 
    ratio = proposedLogTarget+probOldGivenNew-logTarget-probNewGivenOld

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad, {"accept" => true}))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)
      G, invG = copy(proposedG), copy(proposedInvG)
      firstTerm, secondTerm = copy(proposedFirstTerm), copy(proposedSecondTerm)
      if isa(sampler.tuner, EmpMCTuner); tune.accepted += 1; end
    else
      produce(MCMCSample(pars, logTarget, grad, pars, logTarget, grad, {"accept" => false}))
    end

    if isa(sampler.tuner, EmpMCTuner) && i<= runner.burnin && mod(i, sampler.tuner.adaptStep) == 0
      adapt!(tune, sampler.tuner)
      reset!(tune)
      if sampler.tuner.verbose
        println("Burn-in teration $i of $(runner.burnin): ", round(100*tune.rate, 2), " % acceptance rate")
      end
    end

    i += 1
  end
end
