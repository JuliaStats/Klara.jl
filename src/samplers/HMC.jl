###########################################################################
#  Hamiltonian Monte Carlo (HMC)
#
#  Parameters :
#    - nLeaps : number of intermediate jumps wihtin each step
#    - leapStep : inner steps scaling
#
###########################################################################

export HMC

println("Loading HMC(nLeaps, leapStep) sampler")

# The HMC sampler type
immutable HMC <: MCMCSampler
  nLeaps::Integer
  leapStep::Float64

  function HMC(i::Integer, s::Real)
    assert(i>0, "inner steps should be > 0")
    assert(s>0, "inner steps scaling should be > 0")
    new(i,s)
  end
end
HMC() = HMC(10, 0.1)
HMC(i::Integer) = HMC(i, 0.1)
HMC(s::Float64) = HMC(10, s)

# sampling task launcher
spinTask(model::MCMCModel, s::HMC) = MCMCTask( Task(() -> HMCTask(model, s.nLeaps, s.leapStep)), model)

####### HMC sampling

# helper functions and types
type HMCSample
  pars::Vector{Float64} # sample position
  grad::Vector{Float64} # gradient
  v::Vector{Float64}    # velocity
  logTarget::Float64    # log likelihood 
  H::Float64            # Hamiltonian
end
HMCSample(pars::Vector{Float64}) = HMCSample(pars, Float64[], Float64[], NaN, NaN)

calc!(s::HMCSample, ll::Function) = ((s.logTarget, s.grad) = ll(s.pars))
update!(s::HMCSample) = (s.H = s.logTarget - dot(s.v, s.v)/2)

function leapFrog(s::HMCSample, ve, ll::Function)
  n = deepcopy(s)  # make a full copy
  n.v += n.grad * ve / 2.
  n.pars += ve * n.v
  calc!(n, ll)
  n.v += n.grad * ve / 2.
  update!(n)

  n
end


#  HMC algo
function HMCTask(model::MCMCModel, isteps::Integer, leapStep::Float64)
  local state0

  # Task reset function
  function reset(resetPars::Vector{Float64})
    state0 = HMCSample(copy(resetPars))
    calc!(state0, model.evalg)
  end
  # hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 

  # initialization
  state0 = HMCSample(copy(model.init))
  calc!(state0, model.evalg)
  assert(isfinite(state0.logTarget), "Initial values out of model support, try other values")

  #  main loop
  while true
    local j, state

    state0.v = randn(model.size)
    update!(state0)
    state = state0

    j=1
    while j <= isteps && isfinite(state.logTarget)
      state = leapFrog(state, leapStep, model.evalg)
      j +=1
    end

    # accept if new is good enough
    if rand() < exp(state.H - state0.H)
      produce(MCMCSample(state.pars, state.logTarget, state0.pars, state0.logTarget))
      state0 = state
    else
      produce(MCMCSample(state0.pars, state0.logTarget, state0.pars, state0.logTarget))
    end
  end
end

