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

type EmpiricalMALATune
  driftStep::Float64
  accepted::Int
  proposed::Int
  rate::Float64

  function EmpiricalMALATune(driftStep::Float64, accepted::Int, proposed::Int, rate::Float64)
    @assert driftStep > 0 "Leapfrog step size ($driftStep) should be > 0"
    @assert 0 <= accepted "Number of accepted Monte Carlo steps ($accepted) should be non negative"
    @assert 0 <= proposed "Number of proposed Monte Carlo steps ($proposed) should be non negative" 
    new(driftStep, accepted, proposed)
  end
end

EmpiricalMALATune(driftStep::Float64, accepted::Int, proposed::Int) =
  EmpiricalMALATune(driftStep::Float64, accepted::Int, proposed::Int, NaN)

function adapt!(tune::EmpiricalMALATune, tuner::EmpMCTuner)
  tune.rate = tune.accepted/tune.proposed      
  tune.driftStep *= (1/(1+exp(-11*(tune.rate-tuner.targetRate)))+0.5)
end

count!(tune::EmpiricalMALATune) = (tune.accepted += 1)

reset!(tune::EmpiricalMALATune) = ((tune.accepted, tune.proposed) = (0, 0))

###########################################################################
#  MALA type
###########################################################################

# The MALA sampler type
immutable MALA <: MCMCSampler
  driftStep::Float64
  tuner::Union(Nothing, MCMCTuner)
  
  function MALA(s::Real, t::Union(Nothing, MCMCTuner))
    @assert s>0 "MALA drift step should be > 0"
    new(s, t)
  end
end

MALA(s::Float64=1.0) = MALA(s, nothing)
MALA(s::MCMCTuner) = MALA(1.0, t)
MALA(;scale::Float64=1.0, tuner::Union(Nothing, MCMCTuner)=nothing) = MALA(scale, tuner)

# MALA sampling
function SamplerTask(model::MCMCModel, sampler::MALA, runner::MCMCRunner)
  local pars, proposedPars, parsMean
  local logTarget, proposedLogTarget
  local grad, proposedGrad
  local probNewGivenOld, probOldGivenNew
  local driftStep

  @assert hasgradient(model) "MALA sampler requires model with gradient function"

  #  Task reset function
  function reset(resetPars::Vector{Float64})
    pars = copy(resetPars)
    logTarget, grad = model.evalallg(pars)
  end
  # hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 
  
  # Initialization
  pars = copy(model.init)
  logTarget, grad = model.evalallg(pars)
  @assert isfinite(logTarget) "Initial values out of model support, try other values"

  if isa(sampler.tuner, EmpMCTuner); tune = EmpiricalMALATune(sampler.driftStep, 0, 0); end

  i = 1
  while true
    if isa(sampler.tuner, EmpMCTuner)
      tune.proposed += 1
      driftStep = tune.driftStep
    else
      driftStep = sampler.driftStep
    end

    parsMean = pars + (driftStep/2.) * grad

    proposedPars = parsMean + sqrt(driftStep) * randn(model.size)
    proposedLogTarget, proposedGrad = model.evalallg(proposedPars)

    probNewGivenOld = sum(-(parsMean-proposedPars).^2 / (2*driftStep) .- log(2*pi*driftStep)/2)
    parsMean = proposedPars + (driftStep/2) * proposedGrad
    probOldGivenNew = sum(-(parsMean-pars).^2/(2*driftStep) .- log(2*pi*driftStep)/2)
    
    ratio = proposedLogTarget + probOldGivenNew - logTarget - probNewGivenOld
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
