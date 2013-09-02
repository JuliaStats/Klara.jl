###########################################################################
#  Riemannian manifold Hamiltonian Monte Carlo (RMHMC)
#  The RMHMC sampler is work in progress (not running yet)
#
#  Parameters :
#    - nLeaps : number of leapfrog steps
#    - leapStep : leapfrog step size
#    - nNewton: number of Newton steps
#    - tuner: for tuning the RMHMC parameters
#
###########################################################################

export RMHMC

println("Loading RMHMC(nLeaps, leapStep, nNewton, tuner) sampler")

abstract RMHMCTuner <: MCMCTuner

# The RMHMC sampler type
immutable RMHMC <: MCMCSampler
  nLeaps::Integer
  leapStep::Float64
  nNewton::Integer
  tuner::Union(Nothing, RMHMCTuner)
  
  function RMHMC(nLeaps::Integer, leapStep::Float64, nNewton::Integer, tuner::Union(Nothing, RMHMCTuner))
    assert(nLeaps>0, "Number of leapfrog steps should be > 0")
    assert(leapStep>0, "Leapfrog step size should be > 0")
    assert(nNewton>0, "Number of Newton steps should be > 0")    
    new(nLeaps, leapStep, nNewton, tuner)
  end
end

RMHMC(tuner::Union(Nothing, RMHMCTuner)=nothing) = RMHMC(6, 0.5, 4, tuner)
RMHMC(nLeaps::Integer, (tuner::Union(Nothing, RMHMCTuner)=nothing) = RMHMC(nLeaps, 3/nLeaps, 4, tuner)
RMHMC(nLeaps::Integer, leapStep::Float64, (tuner::Union(Nothing, RMHMCTuner)=nothing) = RMHMC(nLeaps, leapStep, 4, tuner)
RMHMC(nLeaps::Integer, nNewton::Integer, (tuner::Union(Nothing, RMHMCTuner)=nothing) = RMHMC(nLeaps, 3/nLeaps, nNewton, tuner)
RMHMC(leapStep::Float64, (tuner::Union(Nothing, RMHMCTuner)=nothing) = RMHMC(int(floor(3/leapStep)), leapStep, 4, tuner)
RMHMC(leapStep::Float64, nNewton::Integer, (tuner::Union(Nothing, RMHMCTuner)=nothing) = RMHMC(int(floor(3/leapStep)), leapStep, nNewton, tuner)

# sampling task launcher
spinTask(model::MCMCModel, s::RMHMC) = MCMCTask(Task(() -> RMHMCTask(model, s.nLeaps, s.leapStep, s.nNewton)), model)

####### RMHMC sampling

# helper functions and types
type RMHMCSample
  pars::Vector{Float64} # sample
  grad::Vector{Float64} # gradient
  tensor::Matrix{Float64} # Metric tensor
  dtensor::Array{Float64, 3} # Derivative of metric tensor
  momentum::Vector{Float64} # momentum
  logtarget::Float64 # log-target 
  H::Float64 # Hamiltonian
end

RMHMCSample(pars::Vector{Float64}) = RMHMCSample(pars, Array(Float64, 0), Array(Float64, 0, 0), Array(Float64, 0, 0, 0), NaN, NaN)

# update!(s::HMCSample) = (s.H = s.logTarget - dot(s.v, s.v)/2)
# 
# function leapFrog(s::HMCSample, ve, ll::Function)
#   n = deepcopy(s)  # make a full copy
#   n.v += n.grad * ve / 2.
#   n.pars += ve * n.v
#   calc!(n, ll)
#   n.v += n.grad * ve / 2.
#   update!(n)
# 
#   n
# end

function SamplerTask(model::MCMCModel, sampler::RMHMC)
  local state0
  
  # hook inside Task to allow remote resetting
  task_local_storage(:reset, (resetPars::Vector{Float64}) -> (state0 = RMHMCSample(copy(resetPars)); calc!(state0, model.evaldt)))
  
  # initialization
  state0 = RMHMCSample(copy(model.init))
  calc!(state0, model.evaldt)
  
  while true
  end
end
