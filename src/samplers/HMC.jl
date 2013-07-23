###########################################################################
#  Hamiltonian Monte-Carlo
#
#     parameters :
#        - innerSteps : number of intermediate jumps wihtin each step
#        - stepSize : inner steps scaling
#
###########################################################################

export HMC

println("Loading HMC(innerSteps, stepSize) sampler")

# The HMC sampler type
immutable HMC <: MCMCSampler
  innerSteps::Integer
  stepSize::Float64

  function HMC(i::Integer, s::Real)
    assert(i>0, "inner steps should be > 0")
    assert(s>0, "inner steps scaling should be > 0")
    new(i,s)
  end
end
HMC() = HMC(2, 1.)
HMC(i::Integer) = HMC(i, 1.)
HMC(s::Float64) = HMC(2, s)

# sampling task launcher
spinTask(model::MCMCModelWithGradient, s::HMC) = 
  MCMCTask( Task(() -> HMCTask(model, s.innerSteps, s.stepSize)), model)

####### HMC sampling

# helper functions and types
type HMCSample
  beta::Vector{Float64}   # sample position
  grad::Vector{Float64}   # gradient
  v::Vector{Float64}    # speed
  llik::Float64     # log likelihood 
  H::Float64        # Hamiltonian
end
HMCSample(beta::Vector{Float64}) = HMCSample(beta, Float64[], Float64[], NaN, NaN)

calc!(s::HMCSample, ll::Function) = ((s.llik, s.grad) = ll(s.beta))
update!(s::HMCSample) = (s.H = s.llik - dot(s.v, s.v)/2)

function leapFrog(s::HMCSample, ve, ll::Function)
  n = deepcopy(s)  # make a full copy
  n.v += n.grad * ve / 2.
  n.beta += ve * n.v
  calc!(n, ll)
  n.v += n.grad * ve / 2.
  update!(n)

  n
end


#  HMC algo
function HMCTask(model::MCMCModelWithGradient, isteps::Integer, stepsize::Float64)
  local state0

  #  Task reset function
  function reset(newbeta::Vector{Float64})
    state0 = HMCSample(copy(newbeta))
    calc!(state0, model.evalg)
  end
  # hook inside Task to allow remote resetting
  task_local_storage(:reset, reset) 

  # initialization
  state0 = HMCSample(copy(model.init))
  calc!(state0, model.evalg)
  assert(isfinite(state0.llik), "Initial values out of model support, try other values")

  #  main loop
  while true
    local j, state

    state0.v = randn(model.size)
    update!(state0)
    state = state0

    j=1
    while j <= isteps && isfinite(state.llik)
      state = leapFrog(state, stepsize, model.evalg)
      j +=1
    end

    # accept if new is good enough
    if rand() < exp(state.H - state0.H)
      produce(MCMCSample(state.beta, state.llik, state0.beta, state0.llik))
      state0 = state
    else
      produce(MCMCSample(state0.beta, state0.llik, state0.beta, state0.llik))
    end

  end

end

