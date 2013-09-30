###########################################################################
#  Hamiltonian Monte Carlo (HMC)
#
#  Parameters :
#    - nLeaps : number of intermediate jumps wihtin each step
#    - leapStep : inner steps scaling
#    - tuner: for tuning the HMC parameters
#
###########################################################################

export HMC, HMCDA

println("Loading HMC(nLeaps, leapStep, tuner) sampler")

###########################################################################
#                  HMC specific 'tuners'
###########################################################################
abstract HMCTuner <: MCMCTuner

type HMCDA <: HMCTuner
  g::Float64 # Free parameter that controls the amount of shrinkage towards
  t0::Float64 # Free parameter that stabilizes the initial iterations of the algorithm
  k::Float64 # Parameter for setting step size scheduling (t^-k)
  l::Float64 # Target simulation length
  r::Float64 # Target HMC acceptance rate
  leapStep::Float64
  H::Float64

  function HMCDA(g::Float64, t0::Float64, k::Float64, l::Float64, r::Float64, leapStep::Float64, H::Float64)
    assert(g > 0, "g parameter of HMCDA tuner ($g) must be positive")
    assert(t0 >= 0, "t0 parameter of HMCDA tuner ($t0) must be non-negative")
    assert(l > 0, "l parameter of HMCDA tuner ($l) must be non-negative")
    assert(r > 0. && r < 1., "Target acceptance rate ($r) should be between 0 and 1")
    assert(leapStep > 0, "leapStep parameter of HMCDA tuner must be positive")
    new(g, t0, k, l, r, leapStep, H)
  end
end 

HMCDA(g::Float64, t0::Float64, k::Float64, l::Float64, r::Float64) = HMCDA(g, t0, k, l, r, 1., 0.)
HMCDA(; g::Float64=0.05, t0::Float64=10., k::Float64=0.75, l::Float64=2., r::Float64=0.65) =
  HMCDA(g, t0, k, l, r, 1., 0.)

###########################################################################
#                  HMC type
###########################################################################

immutable HMC <: MCMCSampler
  nLeaps::Integer
  leapStep::Float64
  tuner::Union(Nothing, HMCTuner)

  function HMC(i::Integer, s::Real, t::Union(Nothing, HMCTuner))
    assert(i>0, "inner steps should be > 0")
    assert(s>0, "inner steps scaling should be > 0")
    new(i,s,t)
  end
end
HMC(i::Integer=10, s::Float64=0.1                                      ) = HMC(i , s  , nothing)
HMC(               s::Float64    , t::Union(Nothing, HMCTuner)=nothing ) = HMC(10, s  , t)
HMC(i::Integer   ,                 t::HMCTuner                         ) = HMC(i , 0.1, t)
HMC(                               t::HMCTuner                         ) = HMC(10, 0.1, t)
# keyword args version
HMC(;init::Integer=10, scale::Float64=0.1, tuner::Union(Nothing, HMCTuner)=nothing) = HMC(init, scale, tuner)

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

function initializeLeapStep(model::MCMCModel, tuner::HMCDA, sample::HMCSample)
  epsilon::Float64 = 1

  # Determine towards which direction epsilon will move
  s = leapFrog(sample, epsilon, model.evalallg)
  update!(s)
  p = exp(s.H-sample.H)
  a = 2*(p>0.5)-1

  # Keep moving epsilon in that direction until acceptprob crosses 0.5
  while p^a > 2^(-a)
    epsilon = epsilon*2^a
    s = leapFrog(sample, epsilon, model.evalallg)
    update!(s)
    p = exp(s.H-sample.H)
  end

  return epsilon
end

####  HMC task
function SamplerTask(model::MCMCModel, sampler::HMC, runner::MCMCRunner)
  local state0
  local p
  local nLeaps::Integer
  local leapStep::Float64

  assert(hasgradient(model), "HMC sampler requires model with gradient function")

  # hook inside Task to allow remote resetting
  task_local_storage(:reset,
             (resetPars::Vector{Float64}) -> (state0 = HMCSample(copy(resetPars)); 
                                              calc!(state0, model.evalallg)) ) 

  # initialization
  state0 = HMCSample(copy(model.init))
  calc!(state0, model.evalallg)

  if isa(sampler.tuner, HMCDA)
    state0.m = randn(model.size)
    leapStep = initializeLeapStep(model, sampler.tuner, state0)
    mu = log(10*leapStep)
    leapStepBar = sampler.tuner.leapStep
  else
    leapStep = sampler.leapStep
  end

  #  main loop
  i = 1
  while true
    local j, state

    state0.m = randn(model.size)
    update!(state0)
    state = state0

    if isa(sampler.tuner, HMCDA)
      nLeaps = max(1, round(sampler.tuner.l/leapStep))
    else
      nLeaps = sampler.nLeaps
    end 

    j=1
    while j <= nLeaps && isfinite(state.logTarget)
      state = leapFrog(state, leapStep, model.evalallg)
      j +=1
    end

    # accept if new is good enough
    p = exp(state.H - state0.H)
    if rand() < p
      ms = MCMCSample(state.pars, state.logTarget, state.grad,
                      state0.pars, state0.logTarget, state0.grad,
                      {"accept" => true} )
      produce(ms)
      state0 = state
    else
      ms = MCMCSample(state0.pars, state0.logTarget, state0.grad,
                      state0.pars, state0.logTarget, state0.grad,
                      {"accept" => false} )
      produce(ms)
    end

    if isa(sampler.tuner, HMCDA)
      if 2 <= i <= runner.burnin
        eta = 1/(i-1+sampler.tuner.t0)
        sampler.tuner.H = (1-eta)*sampler.tuner.H+eta*(sampler.tuner.r-p)
        leapStep = exp(mu - sqrt(i-1)/sampler.tuner.g*sampler.tuner.H)
        eta = (i-1)^-sampler.tuner.k
        leapStepBar = exp((1-eta)*log(leapStepBar)+eta*log(leapStep))
      else
        leapStep = leapStepBar
      end
    end
    i += 1 
  end
end
