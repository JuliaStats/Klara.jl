abstract MCModel

### Sampler types hold the components that fully specify a Monte Carlo sampler
### The fields of sampler types are user-inputed

abstract MCSampler

abstract MHSampler <: MCSampler # Family of samplers based on Metropolis-Hastings
abstract HMCSampler <: MCSampler # Family of Hamiltonian Monte Carlo samplers
abstract LMCSampler <: MCSampler # Family of Langevin Monte Carlo samplers

### Runner types indicate what type of simulation will be run (ex serial or sequential Monte Carlo)
### Their fields fully specify the simulatin details (ex total number or number of burn-in iterations)

abstract ParallelismLevel
type Serial <: ParallelismLevel end
type Parallel <: ParallelismLevel end

abstract MCRunner{P<:ParallelismLevel}

typealias SerialMCRunner MCRunner{Serial}
typealias ParrallelMCRunner MCRunner{Parallel}

### Tuner types

abstract MCTuner

### Monte Carlo sample types hold a single Monte Carlo sample
### A typical MCSample includes at least the vector of parameter values and the respective log-target value

abstract DerivativeOrder
type NullOrder <: DerivativeOrder end
type FirstOrder <: DerivativeOrder end
type SecondOrder <: DerivativeOrder end
type ThirdOrder <: DerivativeOrder end

abstract MCSample{O<:DerivativeOrder}

### Tune types hold the temporary output of the sampler that is used for tuning the sampler

abstract MCTune

# MCState holds two samples at each Monte Carlo iteration, the current and the successive one

type MCState{S<:MCSample}
  successive::S # If proposed sample is accepted, then successive = proposed, otherwise successive = current
  current::S
  diagnostics::Dict
end

MCState{S<:MCSample}(p::S, c::S) = MCState(p, c, Dict())
MCState{S<:MCSample}(p::S) = MCState(p, S(), Dict())

### Stash types hold the temporary components used by a Monte Carlo sampler during its run
### This means that stash types represent the internal state ("local variables") of a Monte Carlo sampler

abstract MCStash{S<:MCSample}

### Monte Carlo Jobs (ex plain job, job using tasks and MPI jobs)

abstract MCTask

abstract MCJob

### Monte Carlo system gathers all the components that define a Monte Carlo simulation
### It is in a sense as a complete specififcation of a Monte Carlo simulation based on user input
### The user mainly interacts with the MCSystem type at a higher level via the package's interface
### Users familiar with the package can also interact with the MCSystem type directly

type MCSystem
  model::MCModel
  sampler::MCSampler
  runner::MCRunner
  tuner::MCTuner
  job::MCJob
end
