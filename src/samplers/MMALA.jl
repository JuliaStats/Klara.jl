###########################################################################
#  Manifold Metropolis adjusted Langevin algorithm (MMALA)
#  MMALA has been deprecated and succeeded by PMALA
#
#  Reference: Girolami M, Calderhead B. Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods. Journal of the
#  Royal Statistical Society: Series B (Statistical Methodology), 2011, 73 (2), pp 123–214
#
#  Parameters :
#    - driftStep : drift step size (for scaling the jumps)
#    - tuner: for tuning the drift step size
#
###########################################################################

export MMALA

println("Loading MMALA(driftStep, tuner) sampler")

###########################################################################
# MMALA specific tuners
###########################################################################
abstract MMALATuner <: MCMCTuner

###########################################################################
# MMALA type
###########################################################################
immutable MMALA <: MCMCSampler
  driftStep::Float64
  tuner::Union(Nothing, MCMCTuner)
  
  function MMALA(s::Real, t::Union(Nothing, MCMCTuner))
    @assert s>0 "MMALA drift step should be > 0"
    new(s, t)
  end
end

MMALA(s::Float64=1.0) = MMALA(s, nothing)
MMALA(s::MALATuner) = MMALA(1.0, t)
MMALA(;scale::Float64=1.0, tuner::Union(Nothing, MCMCTuner)=nothing) = MMALA(scale, tuner)

###########################################################################
# MMALA sampler
###########################################################################
function SamplerTask(model::MCMCModel, sampler::MMALA, runner::MCMCRunner)
  # Define local variables
  local pars, proposedPars, parsMean
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local H, proposedH
  local G, invG, cholOfInvG, dG, invGxdG, traceInvGxdG, proposedG, proposedInvG
  local firstTerm, secondTerm, thirdTerm, proposedFirstTerm, proposedSecondTerm, proposedThirdTerm
  local probNewGivenOld, probOldGivenNew
  local driftStep

  @assert hasgradient(model) "MMALA sampler requires model with gradient function"
  @assert hastensor(model) "MMALA sampler requires model with tensor function"
  @assert hasdtensor(model) "MMALA sampler requires model with function of tensor derivatives"

  # Allocate memory for some of the local variables
  invGxdG = Array(Float64, model.size, model.size, model.size)
  traceInvGxdG = Array(Float64, model.size)
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
    traceInvGxdG[i] = trace(invGxdG[:, :, i])
    secondTerm[:, i] = invGxdG[:, :, i]*invG[:, i]
  end
  thirdTerm = invG*traceInvGxdG
 
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
    parsMean = (pars+(driftStep/2)*firstTerm-driftStep*sum(secondTerm, 2)[:]
      +(driftStep/2)*thirdTerm)

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
      traceInvGxdG[j] = trace(invGxdG[:, :, j])
      proposedSecondTerm[:, j] = invGxdG[:, :, j]*proposedInvG[:, j]
    end
    proposedThirdTerm = proposedInvG*traceInvGxdG

    # Calculate the drift term
    parsMean = (proposedPars+(driftStep/2)*proposedFirstTerm
      -driftStep*sum(proposedSecondTerm, 2)[:]+(driftStep/2)*proposedThirdTerm)

    probOldGivenNew = (-sum(log(diag(chol(driftStep*eye(model.size)*proposedInvG))))
      -(0.5*(parsMean-pars)'*(proposedG/driftStep)*(parsMean-pars))[1])
 
    ratio = proposedLogTarget+probOldGivenNew-logTarget-probNewGivenOld

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad, {"accept" => true}))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)
      G, invG = copy(proposedG), copy(proposedInvG)
      firstTerm, secondTerm, thirdTerm = copy(proposedFirstTerm), copy(proposedSecondTerm), copy(proposedThirdTerm)
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
