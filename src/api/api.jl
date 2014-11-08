typealias FunctionOrNothing Union(Function, Nothing)
typealias F64OrVectorF64 Union(Float64, Vector{Float64})

abstract MCModel

### Sampler types hold the components that fully specify a Monte Carlo sampler
### The fields of sampler types are user-inputed

abstract MCSampler

abstract MHSampler <: MCSampler # Family of samplers based on Metropolis-Hastings
abstract HMCSampler <: MCSampler # Family of Hamiltonian Monte Carlo samplers
abstract LMCSampler <: MCSampler # Family of Langevin Monte Carlo samplers

### Monte Carlo sample types hold a single Monte Carlo sample
### A typical MCSample includes at least the vector of parameter values and the respective log-target value

abstract DerivativeOrder
type NullOrder <: DerivativeOrder end
type FirstOrder <: DerivativeOrder end
type SecondOrder <: DerivativeOrder end
type ThirdOrder <: DerivativeOrder end

abstract MCSample{O<:DerivativeOrder}

### Runner types indicate what type of simulation will be run (ex serial or sequential Monte Carlo)
### Their fields fully specify the simulation details (ex total number or number of burn-in iterations)

abstract ParallelismLevel
type Serial <: ParallelismLevel end
type Parallel <: ParallelismLevel end

abstract MCRunner{P<:ParallelismLevel}

typealias SerialMCRunner MCRunner{Serial}
typealias ParrallelMCRunner MCRunner{Parallel}

### Tuner types

abstract MCTuner

### Tune types hold the temporary output of the sampler that is used for tuning the sampler

abstract MCTune

### Stash types hold the temporary components used by a Monte Carlo sampler during its run
### This means that stash types represent the internal state ("local variables") of a Monte Carlo sampler

abstract MCStash{S<:MCSample}

### Monte Carlo Jobs (ex plain jobs, jobs using tasks or MPI jobs)

abstract MCJob
