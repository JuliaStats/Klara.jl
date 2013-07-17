MCMC.jl  (Work in progress  !)
=======


Current prototyping directions : 

   - a MCMC chain is produced by a triplet : Model x Sampler x Runner :
   - Model = 1) a function that returns a log-likelihood _optionnally accompanied with a gradient / hessian / whatever a specific sampler might need_ + 2) the dimension of the model parameter vector + 3) initial values for model parameters
   - Sampler = an algorithm that from an intial parameter vector produces another parameter vector, the 'proposal', and can be called repeatedly. The sampler will need to store an internal state (such as variables that are adaptively tuned during the run). It is currently implemented as a Julia Task. The Task will be stored in the result structure (the 'chain') which will allow to restart the chain where it stopped.
   - Runner = at its simplest a loop that calls the sampler n times and stores the successive parameter vectors in the chain structure. For population MCMC, an algorithm that starts several sampling runs and adapts the initial parameter vector of the next step accordingly.


To see how all this plays out, I have coded a simple runner and the Random Walk Metropolis.

update (July 17th) : added MALA and HMC samplers, tweaked syntax + ported all expression parsing and autodiff

```jl

using MCMC

# Model definition, 
#  method 1 = state explictly your functions

mymodel = Model(v-> -dot(v,v), 3, ones(3))  # loglik of Normal distrib, vector of 3, initial values 1.0

# or for a model providing the gradient : 
mymodel2 = ModelG(v-> -dot(v,v), v->(-dot(v,v), -2v), 3, ones(3))  # 2nd function returns a tuple (loglik, gradient)


#  method 2 = using expression parsing and autodiff

modexpr = quote
	v ~ Normal(0, 1)
end

mymodel = Model(modexpr, v=ones(3))  # without gradient
mymodel2 = ModelG(modexpr, v=ones(3))  # with gradient


##### running a single chain

res = mymodel * RWM(0.1) * (100:1000)  # burnin = 99
res.samples  # prints samples

res = res * (1:10000)  # continue sampling where it stopped


mymodel * MALA(0.1) * (1:1000) # throws an error because mymodel 
                               #  does not provide the gradient function MALA sampling needs

mymodel2 * MALA(0.1) * (1:1000) # now this works


##### running multiple chains

res = mymodel2 * [RWM(0.1), MALA(0.1), HMC(3,0.1)] * (1:1000) # test all 3 samplers
res[1].samples  # prints samples
res[2].samples  # prints samples
res[3].samples  # prints samples

res = mymodel2 * [HMC(i,0.1) for i in 1:5] * (1:1000) # test HMC with varying # of inner steps



```
