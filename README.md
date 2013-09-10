MCMC.jl
=======

# Scope

The Julia MCMC package provides a generic engine for implementing Bayesian statistical models using Markov Chain Monte
Carlo (MCMC) methods. While the package's framework aims at being extensible to acoommodate user-specific models and
sampling algorithms, it ships with a wide selection of built-in MCMC samplers. It further offers output analysis and
diagnostic MCMC tools.

# Prototype

Current prototyping directions : 

- a MCMC chain is produced by a triplet : Model x Sampler x Runner :
- Model = 1) a function that returns a log-likelihood _optionnally accompanied with a gradient / hessian / whatever a specific sampler might need_ + 2) the dimension of the model parameter vector + 3) initial values for model parameters
- Sampler = an algorithm that from an intial parameter vector produces another parameter vector, the 'proposal', and can be called repeatedly. The sampler will need to store an internal state (such as variables that are adaptively tuned during the run). It is currently implemented as a Julia Task. The Task will be stored in the result structure (the 'chain') which will allow to restart the chain where it stopped.
- Runner = at its simplest a loop that calls the sampler n times and stores the successive parameter vectors in the chain structure. For population MCMC, an algorithm that starts several sampling runs and adapts the initial parameter vector of the next step accordingly.

```jl

using MCMC
using DataFrames

######## Model definition / method 1 = state explictly your functions

# loglik of Normal distrib, vector of 3, initial values 1.0
mymodel = model(v-> -dot(v,v), init=ones(3))  

# or for a model providing the gradient : 
mymodel2 = model(v-> -dot(v,v), grad=v->-2v, init=ones(3))   

######## Model definition / method 2 = using expression parsing and autodiff

modexpr = quote
    v ~ Normal(0, 1)
end

mymodel = model(modexpr, v=ones(3)) # without gradient
mymodel2 = model(modexpr, gradient=true, v=ones(3)) # with gradient

######## running a single chain ########

# Syntax 1 / RWM sampler, burnin = 100, keeping iterations 101 to 1000
res = run(mymodel * RWM(0.1), steps=1000, burnin=100)

# Syntax 1 / RWM sampler, burnin = 100, keeping every 5th iteration between 101 and 1000 
#  (180 post-burnin points)
res = run(mymodel * RWM(0.1), steps=1000, burnin=100, thinning=5)

# Syntax 2 / using a Range for argument 'step'
res = run(mymodel * RWM(0.1), steps=101:5:1000)

# Syntax 3 / using the '*' operator instead of 'run()'
res = mymodel * RWM(0.1) * (101:5:1000)  

# prints samples
head(res.samples)
describe(res.samples)

# prints run diagnostics
head(res.diagnostics)


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
using DataFrames
using Distributions
using MCMC
using Vega

# We need to define a set of models that converge toward the 
#  distribution of interest (in the spirit of simulated annealing)
nmod = 10  # number of models
mods = Array(MCMCLikModel, nmod)
sts = logspace(1, -1, nmod)
for i in 1:nmod
	m = quote
		y = abs(x)
		y ~ Normal(1, $(sts[i]) )
	end
	mods[i] = model(m, x=0)
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
res = seqMC(targets, particles, steps=10, burnin=0)  

# Plot a subset of raw samples
ts = collect(1:10:nrow(res.samples))
plot(x = ts, y = res.samples[ts, "x"], kind = :scatter)
# we don't have the real distribution yet because we didn't use the 
#   sample weightings sequential MC produces

# Now resample with replacement using weights
newsamp = wsample(res.samples["x"], res.diagnostics["weigths"], 1000)

mean(newsamp)  # close to 0 ?
plot(x = [1:1000], y = newsamp, kind = :scatter)

```



