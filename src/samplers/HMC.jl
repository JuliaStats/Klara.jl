###########################################################################
#  Hamiltonian Monte Carlo (HMC)
#
#  Parameters :
#    - nLeaps : number of intermediate jumps wihtin each step
#    - leapStep : inner steps scaling
#    - tuner: for tuning the HMC parameters
#
###########################################################################

export HMC

println("Loading HMC(nLeaps, leapStep, tuner) sampler")

###########################################################################
#                  HMC specific 'tuners'
###########################################################################
abstract HMCTuner <: MCMCTuner

type EmpiricalHMCTune
  nLeaps::Int
  leapStep::Float64
  accepted::Int
  proposed::Int
  rate::Float64

  function EmpiricalHMCTune(nLeaps::Int, leapStep::Float64, accepted::Int, proposed::Int, rate::Float64)
    @assert nLeaps > 0 "Number of leapfrog steps ($nLeaps) should be > 0"
    @assert leapStep > 0 "leapfrog step size ($leapStep) should be > 0"
    @assert 0 <= accepted "Number of accepted Monte Carlo steps ($accepted) should be non negative"
    @assert 0 <= proposed "Number of proposed Monte Carlo steps ($proposed) should be non negative" 
    new(nLeaps, leapStep, accepted, proposed)
  end
end

EmpiricalHMCTune(nLeaps::Int, leapStep::Float64, accepted::Int, proposed::Int) =
  EmpiricalHMCTune(nLeaps::Int, leapStep::Float64, accepted::Int, proposed::Int, NaN)

function adapt!(tune::EmpiricalHMCTune, tuner::EmpMCTuner)
  tune.rate = tune.accepted/tune.proposed      
  tune.leapStep *= (1/(1+exp(-11*(tune.rate-tuner.targetRate)))+0.5)
  tune.nLeaps = min(tuner.maxStep, ceil(tuner.targetPath/tune.leapStep))
end

count!(tune::EmpiricalHMCTune) = (tune.accepted += 1)

reset!(tune::EmpiricalHMCTune) = ((tune.accepted, tune.proposed) = (0, 0))

###########################################################################
#                  HMC type
###########################################################################

immutable HMC <: MCMCSampler
  nLeaps::Int
  leapStep::Float64
  storeLeaps::Bool
  tuner::Union(Nothing, MCMCTuner)

  function HMC(nLeaps::Int, leapStep::Real, storeLeaps::Bool, tuner::Union(Nothing, MCMCTuner))
    @assert nLeaps>0 "inner steps should be > 0"
    @assert leapStep>0 "inner steps scaling should be > 0"
    new(nLeaps, leapStep, storeLeaps, tuner)
  end
end
HMC(tuner::Union(Nothing, MCMCTuner)=nothing) = HMC(10, 0.1, false, tuner)
HMC(nLeaps::Int, tuner::Union(Nothing, MCMCTuner)=nothing) = HMC(nLeaps, 0.1, false, tuner)
HMC(nLeaps::Int, leapStep::Float64, tuner::Union(Nothing, MCMCTuner)=nothing) = HMC(nLeaps, leapStep, false, tuner)
HMC(nLeaps::Int, storeLeaps::Bool, tuner::Union(Nothing, MCMCTuner)=nothing) = HMC(nLeaps, 0.1, storeLeaps, tuner)
HMC(leapStep::Float64, tuner::Union(Nothing, MCMCTuner)=nothing) = HMC(10, leapStep, false, tuner)
HMC(leapStep::Float64, storeLeaps::Bool, tuner::Union(Nothing, MCMCTuner)=nothing) =
  HMC(10, leapStep, storeLeaps, tuner)
HMC(storeLeaps::Bool, tuner::Union(Nothing, MCMCTuner)=nothing) = HMC(10, 0.1, storeLeaps, tuner)
HMC(;init::Int=10, scale::Float64=0.1, leaps::Bool=false, tuner::Union(Nothing, MCMCTuner)=nothing) =
  HMC(init, scale, leaps, tuner)

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
update!(s::HMCSample) = (s.H = -s.logTarget+0.5*dot(s.m, s.m))

function leapfrog(s::HMCSample, ve, ll::Function)
  n = deepcopy(s)  # make a full copy
  n.m += 0.5*n.grad*ve
  n.pars += ve * n.m
  calc!(n, ll)
  n.m += 0.5*n.grad*ve
  update!(n)

  n
end


####  HMC task
function SamplerTask(model::MCMCModel, sampler::HMC, runner::MCMCRunner)
  local state0
  local nLeaps, leapStep
  local leapState0, leapStates
  
  @assert hasgradient(model) "HMC sampler requires model with gradient function"

  # hook inside Task to allow remote resetting
  task_local_storage(:reset,
             (resetPars::Vector{Float64}) -> (state0 = HMCSample(copy(resetPars)); 
                                              calc!(state0, model.evalallg)) ) 

  # initialization
  state0 = HMCSample(copy(model.init))
  calc!(state0, model.evalallg)
  @assert isfinite(state0.logTarget) "Initial values out of model support, try other values"

  if isa(sampler.tuner, EmpMCTuner); tune = EmpiricalHMCTune(sampler.nLeaps, sampler.leapStep, 0, 0); end

  #  main loop
for i in 1:Inf
    local j, state

    if isa(sampler.tuner, EmpMCTuner)
      tune.proposed += 1
      nLeaps, leapStep = tune.nLeaps, tune.leapStep
    else
      nLeaps, leapStep = sampler.nLeaps, sampler.leapStep
    end

    state0.m = randn(model.size)
    update!(state0)
    state = deepcopy(state0)

    if !sampler.storeLeaps
      for j = 1:nLeaps
        state = leapfrog(state, leapStep, model.evalallg)
      end
    else
      leapStates = Array(HMCSample, nLeaps+1)
      leapStates[1] = deepcopy(state0)
      for j = 2:nLeaps+1
        state = leapfrog(state, leapStep, model.evalallg)
        leapStates[j] = deepcopy(state)
      end
    end

    # accept if new is good enough
    if rand() < exp(state0.H-state.H)
      diagnostics = (!sampler.storeLeaps ? {"accept" => true} : {"accept" => true, "leaps" => leapStates})
      ms = MCMCSample(state.pars, state.logTarget, state.grad, state0.pars, state0.logTarget, state0.grad, diagnostics)
      produce(ms)
      state0 = deepcopy(state)
      if isa(sampler.tuner, EmpMCTuner); tune.accepted += 1; end
    else
      diagnostics = (!sampler.storeLeaps ? {"accept" => false} : {"accept" => false, "leaps" => leapStates})
      ms = MCMCSample(state0.pars, state0.logTarget, state0.grad, state0.pars, state0.logTarget, state0.grad,
        diagnostics)
      produce(ms)
    end

    if isa(sampler.tuner, EmpMCTuner) && i<= runner.burnin && mod(i, sampler.tuner.adaptStep) == 0
      adapt!(tune, sampler.tuner)
      reset!(tune)
      if sampler.tuner.verbose
        println("Burn-in teration $i of $(runner.burnin): ", round(100*tune.rate, 2), " % acceptance rate")
      end
    end
  end
end
