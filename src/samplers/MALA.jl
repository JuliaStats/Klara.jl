###########################################################################
#  Metropolis adjusted Langevin algorithm (MALA)
#
#     parameters :
#        - DriftStep : float to scale the jumps
#      (note 1 : I have removed the periodic monitoring, it should be in the runner, TODO)
#      (note 2 : adaptative driftStep should be implemented within the sampler as it is 
#       sampler specific most of the time, adaptation parameters could be specified in
#        the MALA type though)
#
###########################################################################

export MALA

println("Loading MALA(driftStep, monitorPeriod) sampler")

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
spinTask(model::MCMCModelWithGradient, s::MALA) = 
  MCMCTask( Task(() -> MALATask(model, s.driftStep)), model)

# MALA sampling
function MALATask(model::MCMCModelWithGradient, driftStep::Float64)
  local beta1, grad1, betam
  local probNewGivenOld, probOldGivenNew
  local beta, ll, grad
  local oldbeta, oldll

  #  Task reset function
  function reset(newbeta::Vector{Float64})
    beta = newbeta
    ll, grad = model.evalg(beta)
  end
  # hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 
  
  # Initialization
  beta = copy(model.init)
  ll, grad = model.evalg(beta)
  assert(ll != -Inf, "Initial values out of model support, try other values")

  i = 1
  while true
    betam = beta + (driftStep/2.) * grad

    beta1 = betam + sqrt(driftStep) * randn(model.size)
    ll1, grad1 = model.evalg(beta1)

    probNewGivenOld = sum(-(betam-beta1).^2/(2*driftStep)-log(2*pi*driftStep)/2)
    betam = beta1 + (driftStep/2) * grad1
    probOldGivenNew = sum(-(betam-beta).^2/(2*driftStep)-log(2*pi*driftStep)/2)
    
    ratio = ll1 + probOldGivenNew - ll - probNewGivenOld
    if ratio > 0 || (ratio > log(rand()))  # accepted ?
      produce(MCMCSample(beta1, ll1, beta, ll))
      beta, ll, grad = beta1, ll1, grad1
    else
      produce(MCMCSample(beta, ll, beta, ll))
    end

    i += 1
  end
end

