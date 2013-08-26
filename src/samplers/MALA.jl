###########################################################################
#  Metropolis adjusted Langevin algorithm (MALA)
#
#  Parameters :
#    - driftStep : drift step size (for scaling the jumps)
#
###########################################################################

export MALA

println("Loading MALA(driftStep) sampler")

# The MALA sampler type
immutable MALA <: MCMCSampler
  driftStep::Float64

  function MALA(x::Real)
    assert(x>0, "driftStep should be > 0")
    new(x)
  end
end
MALA() = MALA(1.0)

# sampling task launcher
spinTask(model::MCMCModel, s::MALA) = MCMCTask( Task(() -> MALATask(model, s.driftStep)), model)

# MALA sampling
function MALATask(model::MCMCModel, driftStep::Float64)
  local pars, proposedPars, parsMean
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local probNewGivenOld, probOldGivenNew

  #  Task reset function
  function reset(resetPars::Vector{Float64})
    pars = copy(resetPars)
    logTarget, grad = model.evalg(pars)
  end
  # hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 
  
  # Initialization
  pars = copy(model.init)
  logTarget, grad = model.evalg(pars)
  assert(isfinite(logTarget), "Initial values out of model support, try other values")

  while true
    parsMean = pars + (driftStep/2.) * grad

    proposedPars = parsMean + sqrt(driftStep) * randn(model.size)
    proposedLogTarget, proposedGrad = model.evalg(proposedPars)

    probNewGivenOld = sum(-(parsMean-proposedPars).^2/(2*driftStep)-log(2*pi*driftStep)/2)
    parsMean = proposedPars + (driftStep/2) * proposedGrad
    probOldGivenNew = sum(-(parsMean-pars).^2/(2*driftStep)-log(2*pi*driftStep)/2)
    
    ratio = proposedLogTarget + probOldGivenNew - logTarget - probNewGivenOld
    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, pars, logTarget))
      pars, logTarget, grad = proposedPars, proposedLogTarget, proposedGrad
    else
      produce(MCMCSample(pars, logTarget, pars, logTarget))
    end
  end
end
