###########################################################################
#  Metropolis adjusted Langevin algorithm (MALA)
#
#     parameters :
#        - DriftStep : float to scale the jumps
#        - monitorPeriod : integer indicating how often accept rate should be reported
#                   and the driftStep adapted (not implemented)
#                  FIXME? : acceptRate reporting should be in the runner
#                  FIXME? : driftStep adaptation function should be in the sampler
#
###########################################################################

export MALA

# The MALA sampler type
immutable MALA <: MCMCSampler
  driftStep::Float64
  monitorPeriod::Integer

  function MALA(x::Real, y::Integer)
    assert(x>0, "driftStep should be > 0")
    assert(y>0, "monitorPeriod should be > 0")
    new(x,y)
  end
end
MALA() = MALA(1.0, 1e5)
MALA(x::Float64) = MALA(x, 1e5)


# sampling task launcher
function spinTask(model::MCMCModelWithGradient, s::MALA)

  function MALATask(model::MCMCModelWithGradient, 
                    driftStep::Float64,
                    monitorPeriod::Integer)
    local beta1, grad1, betam
    local probNewGivenOld, probOldGivenNew

    local beta = copy(model.init)
    local ll, grad = model.eval(beta)
    assert(ll != -Inf, "Initial values out of model support, try other values")

    task_local_storage(:reset, (x::Vector{Float64}) -> beta=x)  # hook inside Task to allow remote resetting

    proposed, accepted = 0., 0.

    for i = 1:Inf
      proposed += 1
      
      # beta1 = copy(beta)
      # grad1 = model.evalgradient(beta1)
      # betam = beta1 + (driftStep/2.) * grad1
      betam = beta + (driftStep/2.) * grad

      beta1 = betam + sqrt(driftStep) * randn(model.size)
      ll1, grad1 = model.eval(beta1)

      probNewGivenOld = sum(-(betam-beta1).^2/(2*driftStep)-log(2*pi*driftStep)/2)
      betam = beta1 + (driftStep/2) * grad1
      probOldGivenNew = sum(-(betam-beta).^2/(2*driftStep)-log(2*pi*driftStep)/2)
      
      ratio = ll1 + probOldGivenNew - ll - probNewGivenOld
      if ratio > 0 || (ratio > log(rand()))  # accepted ?
        accepted += 1
        beta, ll, grad = beta1, ll1, grad1
      end

      ### monitor every n steps
      if mod(i, monitorPeriod) == 0
        acceptRate = accepted / proposed
        # driftStep = opts.setDriftStep(i, acceptRate, steps, burnin, driftStep)
        println("Iteration $i of $(steps): ", round(100*acceptRate, 2), " % acceptance ratio")
        proposed, accepted = 0., 0.
      end

      produce(beta)
    end

  MCMCTask( Task(() -> MALATask(model, s.drifStep, s.monitorPeriod)), model)
end

