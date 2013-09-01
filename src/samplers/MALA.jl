###########################################################################
#  Metropolis adjusted Langevin algorithm (MALA)
#
#  Parameters :
#    - driftStep : drift step size (for scaling the jumps)
#    - tuner: for tuning the drift step size
#
###########################################################################

export MALA

println("Loading MALA(driftStep, tuner) sampler")

###########################################################################
# MALA specific 'tuners'
###########################################################################
abstract MALATuner <: MCMCTuner

###########################################################################
#  MALA type
###########################################################################

# The MALA sampler type
immutable MALA <: MCMCSampler
  driftStep::Float64
  tuner::Union(Nothing, MALATuner)
  
  function MALA(s::Real, t::Union(Nothing, MALATuner))
    assert(s>0, "MALA drift step should be > 0")
    new(s, t)
  end
end

MALA(s::Float64=1.0) = MALA(s, nothing)
MALA(s::MALATuner) = MALA(1.0, t)
MALA(;scale::Float64=1.0, tuner::Union(Nothing, MALATuner)=nothing) = MALA(scale, tuner)

# MALA sampling
function SamplerTask(model::MCMCModel, sampler::MALA)
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
    parsMean = pars + (sampler.driftStep/2.) * grad

    proposedPars = parsMean + sqrt(sampler.driftStep) * randn(model.size)
    proposedLogTarget, proposedGrad = model.evalg(proposedPars)

    probNewGivenOld = sum(-(parsMean-proposedPars).^2/(2*sampler.driftStep)-log(2*pi*sampler.driftStep)/2)
    parsMean = proposedPars + (sampler.driftStep/2) * proposedGrad
    probOldGivenNew = sum(-(parsMean-pars).^2/(2*sampler.driftStep)-log(2*pi*sampler.driftStep)/2)
    
    ratio = proposedLogTarget + probOldGivenNew - logTarget - probNewGivenOld
    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, pars, logTarget))
      pars, logTarget, grad = proposedPars, proposedLogTarget, proposedGrad
    else
      produce(MCMCSample(pars, logTarget, pars, logTarget))
    end
  end
end
