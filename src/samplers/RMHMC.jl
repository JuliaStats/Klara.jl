###########################################################################
#  Riemannian manifold Hamiltonian Monte Carlo (RMHMC)
#
#  Reference: Girolami M, Calderhead B. Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods. Journal of the
#  Royal Statistical Society: Series B (Statistical Methodology), 2011, 73 (2), pp 123–214
#
#  Parameters :
#    - nLeaps : number of leapfrog steps
#    - leapStep : leapfrog step size
#    - nNewton: number of Newton steps
#    - tuner: for tuning the RMHMC parameters
#
###########################################################################

export RMHMC

###########################################################################
# RMHMC specific tuners
###########################################################################
abstract RMHMCTuner <: MCMCTuner

###########################################################################
# RMHMC type
###########################################################################
immutable RMHMC <: MCMCSampler
  nLeaps::Int
  leapStep::Float64
  nNewton::Int
  tuner::Union(Nothing, MCMCTuner)
  
  function RMHMC(nLeaps::Int, leapStep::Float64, nNewton::Int, tuner::Union(Nothing, MCMCTuner))
    @assert nLeaps>0 "Number of leapfrog steps should be > 0"
    @assert leapStep>0 "Leapfrog step size should be > 0"
    @assert nNewton>0 "Number of Newton steps should be > 0"    
    new(nLeaps, leapStep, nNewton, tuner)
  end
end

RMHMC(tuner::Union(Nothing, MCMCTuner)=nothing) = RMHMC(6, 0.5, 4, tuner)
RMHMC(nLeaps::Int, tuner::Union(Nothing, MCMCTuner)=nothing) = RMHMC(nLeaps, 3/nLeaps, 4, tuner)
RMHMC(nLeaps::Int, leapStep::Float64, tuner::Union(Nothing, MCMCTuner)=nothing) = RMHMC(nLeaps, leapStep, 4, tuner)
RMHMC(nLeaps::Int, nNewton::Int, tuner::Union(Nothing, MCMCTuner)=nothing) = RMHMC(nLeaps, 3/nLeaps, nNewton, tuner)
RMHMC(leapStep::Float64, tuner::Union(Nothing, MCMCTuner)=nothing) = RMHMC(int(floor(3/leapStep)), leapStep, 4, tuner)
RMHMC(leapStep::Float64, nNewton::Int, tuner::Union(Nothing, MCMCTuner)=nothing) = 
  RMHMC(int(floor(3/leapStep)), leapStep, nNewton, tuner)

###########################################################################
# RMHMC sampler
###########################################################################
function SamplerTask(model::MCMCModel, sampler::RMHMC, runner::MCMCRunner)
  # Define local variables
  local pars, proposedPars, leapPars
  local logTarget, proposedLogTarget
  local grad, proposedGrad, leapGrad
  local H, proposedH
  local G, invG, cholG, dG, invGxdG, traceInvGxdG
  local momentum, leapMomentum, momentumTerm, invGMomentum01, invGMomentum02
  local timeStep, nRandomLeaps
  local nLeaps, leapStep

  @assert hasgradient(model) "RMHMC sampler requires model with gradient function"
  @assert hastensor(model) "RMHMC sampler requires model with tensor function"
  @assert hasdtensor(model) "RMHMC sampler requires model with function of tensor derivatives"

  # Allocate memory for some of the local variables
  invGxdG = Array(Float64, model.size, model.size, model.size)
  traceInvGxdG = Array(Float64, model.size)
  momentumTerm = Array(Float64, model.size)

   #  Task reset function
  function reset(resetPars::Vector{Float64})
    pars = copy(resetPars)
    logTarget, grad, G, dG = model.evalalldt(pars)
  end
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 

  # Initialization
  pars = copy(model.init)
  logTarget, grad = model.evalallg(pars)
  @assert isfinite(logTarget) "Initial values out of model support, try other values"
    
  if isa(sampler.tuner, EmpMCTuner); tune = EmpiricalHMCTune(sampler.nLeaps, sampler.leapStep, 0, 0); end

  i = 1
  while true    
    if isa(sampler.tuner, EmpMCTuner)
      tune.proposed += 1
      nLeaps, leapStep = tune.nLeaps, tune.leapStep
    else
      nLeaps, leapStep = sampler.nLeaps, sampler.leapStep
    end

    proposedPars = copy(pars)

    # Calculate metric tensor G, its inverse invG and its Cholesky factor cholG
    G = model.evalt(proposedPars)
    invG = inv(G)
    cholG = chol(G)

    # Calculate Hamiltonian
    momentum = cholG'*randn(model.size)
    # 0.5*(log(2)+model.size*log(pi)+2*sum(log(diag(cholG)))) is the log determinant of metric tensor G
    H = -logTarget+0.5*(log(2)+model.size*log(pi)+2*sum(log(diag(cholG))))+(momentum'*(invG*momentum))/2

    # Calculate derivative dG of metric tensor G, as well ass invG*dG and trace(invG*dG)
    dG = model.evaldt(proposedPars)
    for j = 1:model.size   
      invGxdG[:, :, j] = invG*dG[:, :, j]
      traceInvGxdG[j] = trace(invGxdG[:, :, j])
    end

    # Perform leapfrog steps
    timeStep = (randn() > 0.5 ? 1. : -1.)
    nRandomLeaps = ceil(rand()*nLeaps)
    
    for j = 1:nRandomLeaps
      leapGrad = model.evalg(proposedPars)
      leapMomentum = copy(momentum)
      for k = 1:sampler.nNewton
        invGMomentum01 = invG*leapMomentum
        for r = 1:model.size
          momentumTerm[r] = (0.5*(leapMomentum'*invGxdG[:, :, r]*invGMomentum01))[1]
        end
        leapMomentum = (momentum+timeStep*(leapStep/2)*(leapGrad-0.5*traceInvGxdG+momentumTerm))
      end

      momentum = copy(leapMomentum)
      invGMomentum02 = invG*momentum
      leapPars = copy(proposedPars)
      for k = 1:sampler.nNewton
        G = model.evalt(leapPars)
        invGMomentum01 = G\momentum
        leapPars = proposedPars+(timeStep*(leapStep/2))*(invGMomentum01+invGMomentum02)
      end
      
      proposedPars = copy(leapPars)
      G = model.evalt(proposedPars)
      invG = inv(G)
      dG = model.evaldt(proposedPars)
      for k = 1:model.size
        invGxdG[:, :, k] = invG*dG[:, :, k]
        traceInvGxdG[k] = trace(invGxdG[:, :, k])
      end
      invGMomentum01 = invG*momentum
      for k = 1:model.size
        momentumTerm[k] = (0.5*(momentum'*invGxdG[:, :, k]*invGMomentum01))[1]
      end

      proposedGrad = model.evalg(proposedPars)
      momentum = (momentum+timeStep*(leapStep/2)*(proposedGrad-0.5*traceInvGxdG+momentumTerm))
    end
    
    # Calculate log-target based on the proposed parameters
    proposedLogTarget = model.eval(proposedPars)

    # Calculate Hamiltonian based on the proposed parameters
    proposedH =
      -proposedLogTarget+0.5*(log(2)+model.size*log(pi)+2*sum(log(diag(chol(G)))))+(momentum'*(invG*momentum))/2
      
    ratio = (H-proposedH)[1]

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad, {"accept" => true}))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)
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
