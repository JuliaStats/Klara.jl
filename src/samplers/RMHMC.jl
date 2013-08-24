###########################################################################
#  Riemannian manifold Hamiltonian Monte Carlo (RMHMC)
#  The RMHMC sampler is work in progress (not running yet)
#
#  parameters :
#    - nLeaps : number of leapfrog steps
#    - leapStep : leapfrog step size
#    - nNewton: number of Newton steps
#
###########################################################################

export RMHMC

println("Loading RMHMC(nLeaps, leapStep, nNewton) sampler")

# The RMHMC sampler type
immutable RMHMC <: MCMCSampler
  nLeaps::Integer
  leapStep::Float64
  nNewton::Integer

  function RMHMC(nLeaps::Integer, leapStep::Float64, nNewton::Integer)
    assert(nLeaps>0, "Number of leapfrog steps should be > 0")
    assert(leapStep>0, "Leapfrog step size should be > 0")
    assert(nNewton>0, "Number of Newton steps should be > 0")    
    new(nLeaps, leapStep, nNewton)
  end
end

RMHMC() = RMHMC(6, 0.5, 4)
RMHMC(nLeaps::Integer) = RMHMC(nLeaps, 3/nLeaps, 4)
RMHMC(nLeaps::Integer, leapStep::Float64) = RMHMC(nLeaps, leapStep, 4)
RMHMC(nLeaps::Integer, nNewton::Integer) = RMHMC(nLeaps, 3/nLeaps, nNewton)
RMHMC(leapStep::Float64) = RMHMC(int(floor(3/leapStep)), leapStep, 4)
RMHMC(leapStep::Float64, nNewton::Integer) = RMHMC(int(floor(3/leapStep)), leapStep, nNewton)

# sampling task launcher
spinTask(model::MCMCModel, s::RMHMC) = MCMCTask(Task(() -> RMHMCTask(model, s.nLeaps, s.leapStep, s.nNewton)), model)

####### RNHMC sampling

# helper functions and types
type HMCSample
  pars::Vector{Float64}     # sample
  grad::Vector{Float64}     # gradient
  hessian::Matrix{Float64}  # Hessian
  tensor::Array{Float64, 3} # Metric tensor
  v::Vector{Float64}        # velocity
  logtarget::Float64        # log-target 
  H::Float64                # Hamiltonian
end
