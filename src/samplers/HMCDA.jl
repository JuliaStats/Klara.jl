###########################################################################
#  Adaptive Hamiltonian Monte Carlo with dual averaging (HMCDA)
#
#  Reference: Hoffman M.D, Gelman A.The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.
#  arXiv, 2011
#
#  Parameters:
#    - shrinkage: parameter that controls the amount of shrinkage towards mu
#    - t0: parameter that stabilizes the initial iterations of the algorithm
#    - step: parameter for setting step size scheduling (t^-step)
#    - len: target simulation length
#    - rate: target HMC acceptance rate
#
###########################################################################

export HMCDA

###########################################################################
#                  HMC type
###########################################################################

immutable HMCDA <: MCMCSampler
  rate::Float64
  len::Float64
  shrinkage::Float64
  t0::Float64
  step::Float64

  function HMCDA(rate::Float64, len::Float64, shrinkage::Float64, t0::Float64, step::Float64)
    @assert 0. < rate < 1. "Target acceptance rate ($rate) should be between 0 and 1"
    @assert len > 0 "len parameter of HMCDA sampler ($len) must be non-negative"
    @assert shrinkage > 0. "shrinkage parameter of HMCDA sampler ($shrinkage) must be positive"
    @assert t0 >= 0 "t0 parameter of HMCDA sampler ($t0) must be non-negative"

    new(rate, len, shrinkage, t0, step)
  end
end

HMCDA(;rate::Float64=0.65, len::Float64=2., shrinkage::Float64=0.05, t0::Float64=10., step::Float64=0.75) =
  HMCDA(rate, len, shrinkage, t0, step)

###########################################################################
#                  HMCDA task
###########################################################################

####  Helper functions and types for HMCDA sampling task

function initializeHMCDAStep(model::MCMCModel, sample::HMCSample)
  leapStep::Float64 = 1

  # Determine towards which direction leap step will move
  s = leapfrog(sample, leapStep, model.evalallg)
  update!(s)
  p = exp(s.H-sample.H)
  a = 2*(p>0.5)-1

  # Keep moving leap step in that direction until acceptprob crosses 0.5
  while p^a > 2^(-a)
    leapStep = leapStep*2^a
    s = leapfrog(sample, leapStep, model.evalallg)
    update!(s)
    p = exp(s.H-sample.H)
  end

  return leapStep
end

####  HMCDA task
function SamplerTask(model::MCMCModel, sampler::HMCDA, runner::MCMCRunner)
  local state0
  local dualH
  local p, i
  local nLeaps, leapStep, dualLeapStep

  @assert hasgradient(model) "HMCDA sampler requires model with gradient function"

  # hook inside Task to allow remote resetting
  task_local_storage(:reset,
    (resetPars::Vector{Float64}) -> (state0 = HMCSample(copy(resetPars)); calc!(state0, model.evalallg))) 

  # initialization
  state0 = HMCSample(copy(model.init))
  calc!(state0, model.evalallg)
  @assert isfinite(state0.logTarget) "Initial values out of model support, try other values"
  
  state0.m = randn(model.size)
  leapStep = initializeHMCDAStep(model, state0)
  mu = log(10*leapStep)
  dualLeapStep = 1.
  dualH = 0.

  #  main loop
  i = 1
  while true
    local j, state

    state0.m = randn(model.size)
    update!(state0)
    state = deepcopy(state0)

    nLeaps = max(1, round(sampler.len/leapStep))

    for j = 1:nLeaps
      state = leapfrog(state, leapStep, model.evalallg)
    end

    # accept if new is good enough
    p = min(1, exp(state0.H-state.H))
    if rand() < p
      ms = MCMCSample(state.pars, state.logTarget, state.grad, state0.pars, state0.logTarget, state0.grad,
        {"accept" => true})
      produce(ms)
      state0 = deepcopy(state)
    else
      ms = MCMCSample(state0.pars, state0.logTarget, state0.grad, state0.pars, state0.logTarget, state0.grad,
        {"accept" => false})
      produce(ms)
    end

    if i < runner.burnin
      eta = 1/(i+sampler.t0)
      dualH = (1-eta)*dualH+eta*(sampler.rate-p)
      leapStep = exp(mu - sqrt(i)*dualH/sampler.shrinkage)
      eta = i^(-sampler.step)
      dualLeapStep = exp((1-eta)*log(dualLeapStep)+eta*log(leapStep))
    else
      leapStep = dualLeapStep
    end

    i += 1
  end
end
