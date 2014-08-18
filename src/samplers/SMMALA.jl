###########################################################################
#  Simplified manifold Metropolis adjusted Langevin algorithm (SMMALA)
#
#  Parameters :
#    - driftStep : drift step size (for scaling the jumps)
#    - tuner: for tuning the drift step size
#
###########################################################################

export SMMALA

###########################################################################
# SMMALA specific tuners
###########################################################################
abstract SMMALATuner <: MCMCTuner

###########################################################################
# SMMALA type
###########################################################################
immutable SMMALA <: MCMCSampler
  driftStep::Float64
  tuner::Union(Nothing, MCMCTuner)
  
  function SMMALA(s::Real, t::Union(Nothing, MCMCTuner))
    @assert s>0 "SMMALA drift step should be > 0"
    new(s, t)
  end
end

SMMALA(s::Float64=1.0) = SMMALA(s, nothing)
SMMALA(s::MALATuner) = SMMALA(1.0, t)
SMMALA(;scale::Float64=1.0, tuner::Union(Nothing, MCMCTuner)=nothing) = SMMALA(scale, tuner)

###########################################################################
# SMMALA sampler
###########################################################################
function SamplerTask(model::MCMCModel, sampler::SMMALA, runner::MCMCRunner)
  # Define local variables
  local pars, proposedPars, parsMean
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local H, proposedH
  local G, invG, cholOfInvG, proposedG, proposedInvG
  local firstTerm, proposedFirstTerm
  local probNewGivenOld, probOldGivenNew
  local driftStep

  @assert hasgradient(model) "SMMALA sampler requires model with gradient function"
  @assert hastensor(model) "SMMALA sampler requires model with tensor function"

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
  @assert isfinite(logTarget) "Initial values out of model support, try other values"

  invG = inv(G)
  firstTerm = invG*grad
  
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
    parsMean = pars+(driftStep/2)*firstTerm

    # Calculate proposed parameters
    cholOfInvG = chol(driftStep*invG)
    proposedPars = parsMean+cholOfInvG'*randn(model.size)

    # Update model based on the proposed parameters
    proposedLogTarget, proposedGrad, proposedG = model.evalallt(proposedPars)

    probNewGivenOld = (-sum(log(diag(cholOfInvG)))
      -(0.5*(parsMean-proposedPars)'*(G/driftStep)*(parsMean-proposedPars))[1])

    proposedInvG = inv(proposedG)
    proposedFirstTerm = proposedInvG*proposedGrad

    # Calculate the drift term
    parsMean = proposedPars+(driftStep/2)*proposedFirstTerm

    probOldGivenNew = (-sum(log(diag(chol(driftStep*eye(model.size)*proposedInvG))))
      -(0.5*(parsMean-pars)'*(proposedG/driftStep)*(parsMean-pars))[1])
 
    ratio = proposedLogTarget+probOldGivenNew-logTarget-probNewGivenOld

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad, {"accept" => true}))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)
      G, invG, firstTerm = copy(proposedG), copy(proposedInvG), copy(proposedFirstTerm)
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
