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


####  HMC task
function SamplerTask(model::MCMCModel, sampler::HMC, runner::MCMCRunner)
  local state0

  assert(hasgradient(model), "HMC sampler requires model with gradient function")

  # hook inside Task to allow remote resetting
  task_local_storage(:reset,
             (resetPars::Vector{Float64}) -> (state0 = HMCSample(copy(resetPars)); 
                                              calc!(state0, model.evalallg)) ) 

  # initialization
  state0 = HMCSample(copy(model.init))
  calc!(state0, model.evalallg)

  #  main loop
  while true
    local j, state

    state0.m = randn(model.size)
    update!(state0)
    state = state0

    j=1
    while j <= sampler.nLeaps && isfinite(state.logTarget)
      state = leapFrog(state, sampler.leapStep, model.evalallg)
      j +=1
    end

    # accept if new is good enough
    if rand() < exp(state.H - state0.H)
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
  end

end
