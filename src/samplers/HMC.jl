###########################################################################
#  Hamiltonian Monte Carlo (HMC)
#
#  Parameters :
#    - nLeaps : number of intermediate jumps wihtin each step
#    - leapStep : inner steps scaling
#    - tuner: for tuning the HMC parameters
#
###########################################################################

export HMC, EDMCTuner

println("Loading HMC(nLeaps, leapStep, tuner) sampler")

###########################################################################
#                  HMC specific 'tuners'
###########################################################################
abstract HMCTuner <: DMCTuner

immutable EmpiricalDMCTuner <: DMCTuner
  adaptStep::Integer
  maxLeapStep::Integer
  targetPath::Float64
  targetRate::Float64

  function EmpiricalDMCTuner(adaptStep::Integer, maxLeapStep::Integer, targetPath::Float64, targetRate::Float64)
    assert(adaptStep > 0, "Adaptation step size ($adaptStep) should be > 0")
    assert(maxLeapStep > 0, "Adaptation step size ($maxLeapStep) should be > 0")    
    assert(0 < targetRate < 1, "Target acceptance rate ($targetRate) should be between 0 and 1")
    new(adaptStep, maxLeapStep, targetPath, targetRate)
  end
end

typealias EDMCTuner EmpiricalDMCTuner

EDMCTuner(adaptStep::Integer, targetPath::Float64, targetRate::Float64) =
  EDMCTuner(adaptStep, 200, targetPath, targetRate)

EDMCTuner(targetPath::Float64, targetRate::Float64) = EDMCTuner(100, 200, targetPath, targetRate)

EDMCTuner(targetRate::Float64) = EDMCTuner(100, 200, 3.5, targetRate)

type EmpiricalDMCTune
  nLeaps::Integer
  leapStep::Float64
  accepted::Integer
  proposed::Integer

  function EmpiricalDMCTune(nLeaps::Integer, leapStep::Float64, accepted::Integer, proposed::Integer)
    assert(nLeaps > 0, "Number of leapfrog steps ($nLeaps) should be > 0")
    assert(leapStep > 0, "Leapfrog step size ($leapStep) should be > 0")
    assert(0 <= accepted, "Number of accepted Monte Carlo steps ($accepted) should be non negative")
    assert(0 <= proposed, "Number of proposed Monte Carlo steps ($proposed) should be non negative")      
    new(nLeaps, leapStep, accepted, proposed)
  end
end

function adapt!(tune::EmpiricalDMCTune, tuner::EDMCTuner)        
  tune.leapStep *= (1/(1+exp(-11*(tune.accepted/tune.proposed-tuner.targetRate)))+0.5)
  tune.nLeaps = min(tuner.maxLeapStep, ceil(tuner.targetPath/tune.leapStep))
end

count!(tune::EmpiricalDMCTune) = (tune.accepted += 1)

reset!(tune::EmpiricalDMCTune) = ((tune.accepted, tune.proposed) = (0, 0))

###########################################################################
#                  HMC type
###########################################################################

immutable HMC <: MCMCSampler
  nLeaps::Integer
  leapStep::Float64
  tuner::Union(Nothing, DMCTuner)

  function HMC(i::Integer, s::Real, t::Union(Nothing, DMCTuner))
    assert(i>0, "inner steps should be > 0")
    assert(s>0, "inner steps scaling should be > 0")
    new(i,s,t)
  end
end
HMC(i::Integer=10, s::Float64=0.1                                      ) = HMC(i , s  , nothing)
HMC(               s::Float64    , t::Union(Nothing, DMCTuner)=nothing ) = HMC(10, s  , t)
HMC(i::Integer   ,                 t::DMCTuner                         ) = HMC(i , 0.1, t)
HMC(                               t::DMCTuner                         ) = HMC(10, 0.1, t)
# keyword args version
HMC(;init::Integer=10, scale::Float64=0.1, tuner::Union(Nothing, DMCTuner)=nothing) = HMC(init, scale, tuner)

###########################################################################
#                  HMC task
###########################################################################

####  Helper functions and types for HMC sampling task
type HMCSample
  pars::Vector{Float64} # sample position
  grad::Vector{Float64} # gradient
  m::Vector{Float64}    # momentum
  logTarget::Float64    # log likelihood 
  H::Float64            # Hamiltonian
end
HMCSample(pars::Vector{Float64}) = HMCSample(pars, Float64[], Float64[], NaN, NaN)

calc!(s::HMCSample, ll::Function) = ((s.logTarget, s.grad) = ll(s.pars))
update!(s::HMCSample) = (s.H = s.logTarget - dot(s.m, s.m)/2)

function leapFrog(s::HMCSample, ve, ll::Function)
  n = deepcopy(s)  # make a full copy
  n.m += n.grad * ve / 2.
  n.pars += ve * n.m
  calc!(n, ll)
  n.m += n.grad * ve / 2.
  update!(n)

  n
end


####  HMC task
function SamplerTask(model::MCMCModel, sampler::HMC, runner::MCMCRunner)
  local state0
  local nLeaps, leapStep

  assert(hasgradient(model), "HMC sampler requires model with gradient function")

  # hook inside Task to allow remote resetting
  task_local_storage(:reset,
             (resetPars::Vector{Float64}) -> (state0 = HMCSample(copy(resetPars)); 
                                              calc!(state0, model.evalallg)) ) 

  # initialization
  state0 = HMCSample(copy(model.init))
  calc!(state0, model.evalallg)

  if isa(sampler.tuner, EDMCTuner); tune = EmpiricalDMCTune(sampler.nLeaps, sampler.leapStep, 0, 0); end

  #  main loop
  i = 1
  while true
    local j, state

    if isa(sampler.tuner, EDMCTuner)
      tune.proposed += 1
      nLeaps, leapStep = tune.nLeaps, tune.leapStep
    else
      nLeaps, leapStep = sampler.nLeaps, sampler.leapStep
    end

    state0.m = randn(model.size)
    update!(state0)
    state = state0

    j=1
    while j <= nLeaps && isfinite(state.logTarget)
      state = leapFrog(state, leapStep, model.evalallg)
      j +=1
    end

    # accept if new is good enough
    if rand() < exp(state.H - state0.H)
      ms = MCMCSample(state.pars, state.logTarget, state.grad,
                      state0.pars, state0.logTarget, state0.grad,
                      {"accept" => true} )
      produce(ms)
      state0 = state
      if isa(sampler.tuner, EDMCTuner); tune.accepted += 1; end
    else
      ms = MCMCSample(state0.pars, state0.logTarget, state0.grad,
                      state0.pars, state0.logTarget, state0.grad,
                      {"accept" => false} )
      produce(ms)
    end

    if isa(sampler.tuner, EDMCTuner) && mod(i, sampler.tuner.adaptStep) == 0
      adapt!(tune, sampler.tuner)
      reset!(tune)
    end

    i += 1
  end

end
