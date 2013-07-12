###########################################################################
#  Metropolis adjusted Langevin algorithm (MALA)
#
#     takes a scalar as parameter to give a scale to jumps
#
###########################################################################

export MALA

# The MALA sampler type
immutable MALA


end

# sampling task launcher
function spinTask(model::MCMCModel, s::MALA)

  function RWMTask(model::MCMCModel, scale::Float64)
    local beta = copy(model.init)
    local ll = model.eval(beta)
    local oldbeta, oldll, jump

    assert(ll != -Inf, "Initial values out of model support, try other values")

    task_local_storage(:reset, (x::Vector{Float64}) -> beta=x)  # hook inside Task to allow remote resetting

    while true
      jump = randn(model.size) * scale
      oldbeta = copy(beta)
      beta += jump 

      oldll, ll = ll, model.eval(beta) 

      if rand() > exp(ll - oldll) # roll back if rejected
        ll, beta = oldll, oldbeta
      end

      produce(beta)
    end
  end

  MCMCTask( Task(() -> RWMTask(model, s.scale)), model)
end



# Function for running Metropolis adjusted Langevin algorithm (MALA)
function mala(model::Model, opts::MalaOpts)
  proposed, accepted = 0., 0.
  
  beta = model.init  # randPrior() should be called at model creation 
  ll = model.evalll(beta)
  grad = model.evalgradient(beta)
  driftStep = opts.setDriftStep(1, 0., steps, burnin, 0.)

  for i = 1:steps
    proposed += 1
    
    beta1 = copy(beta)
    grad1 = model.evalgradient(beta1)
    betam = beta1 + (driftStep/2)*grad1
    beta1 = betam+sqrt(driftStep)*randn(model.nPars)
    ll1 = model.evalll(beta1)
    probNewGivenOld = sum(-(betam-beta1).^2/(2*driftStep)-log(2*pi*driftStep)/2)
    grad1 = model.evalgradient(beta1)
    betam = beta1+(driftStep/2)*grad1
    probOldGivenNew = sum(-(betam-beta).^2/(2*driftStep)-log(2*pi*driftStep)/2)
    
    ratio = ll1+probOldGivenNew-ll-probNewGivenOld
             
    if ratio > 0 || (ratio > log(rand()))  # accepted ??
      accepted += 1
      beta = copy(beta1)
      ll = copy(ll1)
      grad = copy(grad1)
    end

    ### monitor every n steps
    if mod(i, opts.mcmc.monitorRate) == 0
      acceptRate = accepted/proposed
      driftStep = opts.setDriftStep(i, acceptRate, steps, burnin, driftStep)
      println("Iteration $i of $(steps): ", round(100*acceptRate, 2), " % acceptance ratio")
      proposed, accepted = 0., 0.
    end

  end

  return mcmc, z
end
