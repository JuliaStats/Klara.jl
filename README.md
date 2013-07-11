MCMC.jl  (Work in progress  !)
=======


Current prototyping directions : 

   - a MCMC chain is produced by a triplet : Model x Sampler x Runner :
   - Model = 1) a function that returns a log-likelihood _optionnally accompanied with a gradient / hessian / whatever a specific sampler might need_ + 2) the dimension of the model parameter vector + 3) initial values for model parameters
   - Sampler = an algorithm that from an intial parameter vector produces another parameter vector, the 'proposal', and can be called repeatedly. The sampler will need to store an internal state (such as variables that are adaptively tuned during the run). It is currently implemented as a Julia Task. The Task will be stored in the result structure (the 'chain') which will allow to restart the chain where it stopped.
   - Runner = at its simplest a loop that calls the sampler n times and stores the successive parameter vectors in the chain structure. For population MCMC, an algorithm that starts several sampling runs and adapts the initial parameter vector of the next step accordingly.


To see how all this plays out, I have coded a simple runner and the Random Walk Metropolis.

```jl

using MCMC

# model definition 
#   - with the log-likelihood function (here a normal distribution centered on zero)
#   - a parameter vector of size 3
#   - initial values of 1.0

m = MCMCModel((v)-> -dot(v,v), Dict(), 3, ones(3))

# The simple 'runner' calls the sampling algorithm 'steps' times, 
#  and throws out the first 'burnin' samples
# The sampling algorithm is indicated by the type RMW, instantiated with the settings
# relevant to the sampler (here a scale parameter equal to 1.0)

res = run(m, RWM(1.), steps=10000, burnin=0)

# res is a MCMCChain type containing the samples and the sampling task, allowing to
#  pursue the sampling if needed :

res2 = run(res, steps=10000)

# to simplify the syntax, I have overloaded the getindex method allowing
#  to replace the preceding line with the equivalent : 

res2 = res[1:10000]

# Not apparent in these examples is that the combination of model and sampler
#   creates a MCMCTask structure that can be manipulated separately :

t = m * RWM(1.) # create a MCMCTask

t[100:10000]  # equivalent to steps = 10000 and burnin = 99

# Still playing with * operator, we can define multiple chains
#   which is useful for starting from different initial values or to
#  find the best sampler settings

ts = m * [ RWM(10^x) for x in -2:0.3:-0.2]  # launch multiple RWM chains with different scales

ts[1000:10000]

# or equivalently

run(ts, steps=10000, burnin=999)


```
