MCMC.jl  (Work in progress)
=======


Current prototyping directions : 

   - a MCMC chain is produced by a triplet : Model x Sampler x Runner :
   - Model = 1) a function that returns a log-likelihood _optionnally accompanied with a gradient / hessian / whatever a specific sampler might need_ + 2) the dimension of the model parameter vector + 3) initial values for model parameters
   - Sampler = an algorithm that from an intial parameter vector produces another parameter vector, the 'proposal', and can be called repeatedly. The sampler will need to store an internal state (such as variables that are adaptively tuned during the run). It is currently implemented as a Julia Task. The Task will be stored in the result structure (the 'chain') which will allow to restart the chain where it stopped.
   - Runner = at its simplest a loop that calls the sampler n times and stores the successive parameter vectors in the chain structure. For population MCMC, an algorithm that starts several sampling runs and adapts the initial parameter vector of the next step accordingly.


To see how all this plays out, I have coded a simple runner and the Random Walk Metropolis.

- update (July 17th) : added MALA and HMC samplers, tweaked syntax + ported all expression parsing and autodiff
- update (July 23rd) : added seqMC() a sequential Monte-Carlo runner to see if population MC algorithms can be implemented smoothly in this architecture (see example below) _note that all this is quickly implemented and not thoroughly tested_

```jl

using MCMC
using DataFrames

######## Model definition / method 1 = state explictly your functions

# loglik of Normal distrib, vector of 3, initial values 1.0
mymodel = MCMCLikModel(v-> -dot(v,v), 3, ones(3))  

# or for a model providing the gradient : 
mymodel2 = MCMCLikModelG(v-> -dot(v,v), v->(-dot(v,v), -2v), 3, ones(3))  
# Note that 2nd function returns a tuple (loglik, gradient)

######## Model definition / method 2 = using expression parsing and autodiff

modexpr = quote
	v ~ Normal(0, 1)
end

mymodel = MCMCLikModel(modexpr, v=ones(3))  # without gradient
mymodel2 = MCMCLikModelG(modexpr, v=ones(3))  # with gradient


######## running a single chain ########

# RWM sampler, burnin = 100, keeping iterations 101 to 1000
res = run(mymodel * RWM(0.1), steps=1000, burnin=100)  
# prints samples
head(res.samples)
describe(res.samples)

# continue sampling where it stopped
res = run(res, steps=10000)  


mymodel * MALA(0.1) * (1:1000) # throws an error 
#  ('mymodel' does not provide the gradient function MALA sampling needs)

mymodel2 * MALA(0.1) * (1:1000) # now this works


######## running multiple chains

# test all 3 samplers at once
res = run(mymodel2 * [RWM(0.1), MALA(0.1), HMC(3,0.1)], steps=1000) 
res[2].samples  # prints samples for MALA(0.1)

# test HMC with varying # of inner steps
res = run(mymodel2 * [HMC(i,0.1) for i in 1:5], steps=1000)

```


Now an example with sequential Monte-Carlo

```jl
using MCMC
using Vega

# We need to define a set of models that converge toward the 
#  distribution of interest (in the spirit of simulated annealing)
nmod = 10  # number of models
mods = MCMCLikModel[]
sts = logspace(1, -1, nmod)
for fac in sts
	m = quote
		y = abs(x)
		y ~ Normal(1, $fac )
	end
	push!(mods,MCMCLikModel(m, x=0))
end

# Plot models
xx = linspace(-3,3,100) * ones(nmod)' 
yy = Float64[ mods[j].eval([xx[i,j]]) for i in 1:100, j in 1:nmod]
g = ones(100) * [1:10]'  
plot(x = vec(xx), y = exp(vec(yy)), group= vec(g), kind = :line)

# Build MCMCTasks with diminishing scaling
targets = MCMCTask[ mods[i] * RWM(sts[i]) for i in 1:nmod ]

# Create a 1000 particles
particles = [ [randn()] for i in 1:1000]

# Launch sequential MC 
# (10 steps x 1000 particles = 10000 samples returned in a single MCMCChain)
res = seqMC(targets, particles, steps=10)  

# Plot raw samples
ts = collect(1:10:size(res.samples[:beta],2))
plot(x = ts, y = vec(res.samples[:beta])[ts], kind = :line)
# we don't have the real distribution yet because we didn't use the 
#   sample weightings sequential MC produces

# Now resample with replacement using weights
w = res.misc[:weights]
ns = length(w)
cp = cumsum(w) / sum(w)
rs = fill(0, ns)
for n in 1:ns  #  n = 1
	l = rand()
	rs[n] = findfirst(p-> (p>=l), cp)
end
newsamp = vec(res.samples[:beta])[rs]

mean(newsamp)  # close to 0 ?
plot(x = collect(1:ns), y = newsamp, kind = :scatter)

```



