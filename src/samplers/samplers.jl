### Sampler types hold the components that fully specify a Monte Carlo sampler
### The fields of sampler types are user-inputed

abstract MCSampler

abstract MHSampler <: MCSampler # Family of samplers based on Metropolis-Hastings
abstract HMCSampler <: MCSampler # Family of Hamiltonian Monte Carlo samplers
abstract LMCSampler <: MCSampler # Family of Langevin Monte Carlo samplers

### Stash types hold the temporary components used by a Monte Carlo sampler during its run
### This means that stash types represent the internal state ("local variables") of a Monte Carlo sampler

abstract MCStash
