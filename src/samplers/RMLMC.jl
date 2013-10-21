###########################################################################
#  Semi-explicit Riemannian manifold Lagrangian Monte Carlo (RMLMC)
#
#  Reference: Lan S, Stathopoulos V, Shahbaba B, Girolami M. Lagrangian Dynamical Monte Carlo. arXiv, 2012
#
#  Parameters :
#    - nLeaps : number of leapfrog steps
#    - leapStep : leapfrog step size
#    - nNewton: number of Newton steps
#    - tuner: for tuning the RMLMC parameters
#
###########################################################################

export RMLMC

println("Loading RMLMC(nLeaps, leapStep, nNewton, tuner) sampler")

###########################################################################
# RMLMC specific tuners
###########################################################################
abstract RMLMCTuner <: MCMCTuner

###########################################################################
# RMLMC type
###########################################################################
immutable RMLMC <: MCMCSampler
  nLeaps::Int
  leapStep::Float64
  nNewton::Int
  tuner::Union(Nothing, MCMCTuner)
  
  function RMLMC(nLeaps::Int, leapStep::Float64, nNewton::Int, tuner::Union(Nothing, MCMCTuner))
    @assert nLeaps>0 "Number of leapfrog steps should be > 0"
    @assert leapStep>0 "Leapfrog step size should be > 0"
    @assert nNewton>0 "Number of Newton steps should be > 0"    
    new(nLeaps, leapStep, nNewton, tuner)
  end
end

RMLMC(tuner::Union(Nothing, MCMCTuner)=nothing) = RMLMC(6, 0.5, 4, tuner)
RMLMC(nLeaps::Int, tuner::Union(Nothing, MCMCTuner)=nothing) = RMLMC(nLeaps, 3/nLeaps, 4, tuner)
RMLMC(nLeaps::Int, leapStep::Float64, tuner::Union(Nothing, MCMCTuner)=nothing) = RMLMC(nLeaps, leapStep, 4, tuner)
RMLMC(nLeaps::Int, nNewton::Int, tuner::Union(Nothing, MCMCTuner)=nothing) = RMLMC(nLeaps, 3/nLeaps, nNewton, tuner)
RMLMC(leapStep::Float64, tuner::Union(Nothing, MCMCTuner)=nothing) = RMLMC(int(floor(3/leapStep)), leapStep, 4, tuner)
RMLMC(leapStep::Float64, nNewton::Int, tuner::Union(Nothing, MCMCTuner)=nothing) = 
  RMLMC(int(floor(3/leapStep)), leapStep, nNewton, tuner)

###########################################################################
# RMLMC sampler
###########################################################################
function SamplerTask(model::MCMCModel, sampler::RMLMC, runner::MCMCRunner)
  # Define local variables
  local pars, proposedPars
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local E, proposedE
  local G, proposedG, invG, proposedInvG, cholG, proposedCholG, dG, proposeddG
  local traceInvGxdG, proposedTraceInvGxdG
  local dphi, proposeddphi, C, proposedC
  local velocity, proposedVelocity, initLeapVelocity, vxC, deltaLogDet
  local nRandomLeaps
  local nLeaps, leapStep

  @assert hasgradient(model) "RMLMC sampler requires model with gradient function"
  @assert hastensor(model) "RMLMC sampler requires model with tensor function"
  @assert hasdtensor(model) "RMLMC sampler requires model with function of tensor derivatives"

   #  Task reset function
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
  cholG = chol(G)
  traceInvGxdG = Float64[trace(invG*dG[:, :, i]) for i = 1:model.size]
  dphi = -grad+0.5*traceInvGxdG
  C = 0.5*(permutedims(dG, [3 2 1])+permutedims(dG, [1 3 2])-dG)

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
    proposedG = copy(G)
    proposedInvG = copy(invG)
    proposedCholG = copy(cholG)
    proposeddG = copy(dG)
    proposedTraceInvGxdG = copy(traceInvGxdG)
    proposeddphi = copy(dphi)
    proposedC = copy(C)

    # Sample velocity and calculate current Energy
    velocity = (chol(proposedInvG))'*randn(model.size)
    proposedVelocity = copy(velocity)
    E = -logTarget+sum(log(diag(cholG)))+0.5*velocity'*G*velocity

    vxC = zeros(model.size, model.size)
    # Accumulate determinant to be adjusted in acceptance rate
    deltaLogDet = 0

    # Perform leapfrog steps
    nRandomLeaps = ceil(rand()*nLeaps)

    for j = 1:nRandomLeaps
      # Update velocity
      leapVelocity = copy(proposedVelocity)
      for k = 1:sampler.nNewton
        for r = 1:model.size
          vxC[r, :] = leapVelocity'*proposedC[:, :, r]
        end
        leapVelocity = proposedVelocity-(0.5*leapStep)*proposedInvG*(vxC*leapVelocity+proposeddphi)
      end
      proposedVelocity = copy(leapVelocity)

      #Update volume correction term
      deltaLogDet = deltaLogDet-logdet(proposedG+(leapStep)*vxC)
                
      # Update parameters
      proposedPars = proposedPars+leapStep*proposedVelocity

      proposedLogTarget, proposedGrad, proposedG, proposeddG = model.evalalldt(proposedPars)

      proposedInvG = inv(proposedG)        
      proposedCholG = chol(proposedG)
      proposedTraceInvGxdG = Float64[trace(proposedInvG*proposeddG[:, :, k]) for k = 1:model.size]

      proposeddphi = -proposedGrad+0.5*proposedTraceInvGxdG
      proposedC = 0.5*(permutedims(proposeddG, [3 2 1])+permutedims(proposeddG, [1 3 2])-proposeddG)

      # Update velocity
      for k = 1:model.size
        vxC[k, :] = proposedVelocity'*proposedC[:, :, k]
      end
      deltaLogDet = deltaLogDet+logdet(proposedG-leapStep*vxC)
                
      proposedVelocity = proposedVelocity-(0.5*leapStep)*proposedInvG*(vxC*proposedVelocity+proposeddphi)       
    end
                        
    # Calculate energy based on the proposed parameters
    proposedE = -proposedLogTarget+sum(log(diag(proposedCholG)))+0.5*proposedVelocity'*proposedG*proposedVelocity
                        
    # Accept according to ratio
    ratio = (E-proposedE)[1]+deltaLogDet

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad, {"accept" => true}))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)
      G, dG, invG, cholG = copy(proposedG), copy(proposeddG), copy(proposedInvG), copy(proposedCholG)
      traceInvGxdG, C, dphi = copy(proposedTraceInvGxdG), copy(proposedC), copy(proposeddphi)
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
