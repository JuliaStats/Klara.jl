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

  # TODO 1: complete declaration of local variables

  local pars, proposedPars
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local E, proposedE
  local G, proposedG, invG, proposedInvG, cholG, proposedCholG, dG, proposeddG
  local traceInvGxdG, proposedTraceInvGxdG
  local velocity, proposedVelocity, vxC, deltaLogDet
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

  # TODO 2: complete initialization

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

    # TODO 3: complete copy of variables

    # TODO 4: complete first main part of algorithm

    # Perform leapfrog steps
    nRandomLeaps = ceil(rand()*nLeaps)
    timeStep = (randn() > 0.5 ? 1. : -1.)

    nRandomLeaps = ceil(rand()*nLeaps)

    for j = 1:nRandomLeaps

    # TODO 5: complete second main part of algorithm
            
    # Accept according to ratio
    ratio = (E-proposedE)[1]+deltaLogDet

    if ratio > 0 || (ratio > log(rand()))  # i.e. if accepted
      produce(MCMCSample(proposedPars, proposedLogTarget, proposedGrad, pars, logTarget, grad, {"accept" => true}))
      pars, logTarget, grad = copy(proposedPars), copy(proposedLogTarget), copy(proposedGrad)

      # TODO 6: complete copy of variables

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
