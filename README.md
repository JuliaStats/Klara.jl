## Julia MCMC

The Julia `MCMC` package provides a generic engine for implementing Bayesian statistical models using Markov Chain Monte
Carlo (MCMC) methods. While the package's framework aims at being extensible to accommodate user-specific models and
sampling algorithms, it ships with a wide selection of built-in MCMC samplers. It further offers output analysis and
MCMC diagnostic tools. 

_This package is built on the prior work that went into the `GeometricMCMC` and `SimpleMCMC` packages._

## Features

The core functionality of Julia's `MCMC` package includes:
- various MCMC samplers, up-to-date with contemporary advances in MCMC methodology,
- ability to suspend/resume MCMC simulations flexibly by using Julia tasks,
- user-friendly syntax for model specification,
- use of automatic differentiation to generate the gradient function (higher order derivatives generation is under consideration),
- integration with and use of functionality of DataFrames and Distributions packages,
- post-processing of MCMC output, such as variance reduction methods.

## Main Components of MCMC Chains in Julia

A Monte Carlo chain is produced by a triplet model x sampler x runner:
- The model component refers in general to a Bayesian model on which MCMC inference is performed. Likelihood-based
modelling is currently supported, which typically requires knowledge of the log-likelihood and of the model parameters'
initial values. The derivatives of the log-likelihood (including its gradient, tensor and derivatives of tensor) are
required by some of the MCMC routines and can be optionally specified as functions (or computed by the package's
automatic differentiation algorithms in the case of the gradient).
- The sampler represents the MCMC algorithm. Starting from an intial parameter vector, it produces another parameter
vector, the 'proposal', and can be called repeatedly. The sampler stores an internal state, such as variables that are
adaptively tuned during the run. It is implemented as a Julia `Task`. The task is stored in the result structure, the
'chain', which allows to restart the chain where it stopped.
- The runner is at its simplest a loop that calls the sampler repeatedly and stores the successive parameter vectors in
the chain structure. For population MCMC, the runner spawns several sampling runs and adapts the initial parameter
vector of the next step accordingly.

## Short tutorial

To demonstrate `MCMC`'s usage, two introductory examples are provided. A user guide will be available soon to
elaborate on the package's features in further detail.

### Example 1: Basic syntax for running a single Monte Carlo chain

As a walk-through example, consider three independent identically distributed random variables, each following a Normal
distribution N(0, 1). This implies simulating a Markov chain from the three-dimensional Normal target N(0, I).

#### Model definition

There are two ways to define the target distribution. The first relies on passing the functional definition of the
log-target (and possibly its gradient) as arguments to the `model()` function. Alternatively, one may define the target
as an expression and then pass it to `model()`.

```jl
using MCMC

### First way of defining the model by explicitly defining the log-target

# Log of Normal(0, 1): v-> -dot(v,v), initial values of parameters: (1.0, 1.0, 1.0)
mymodel1 = model(v-> -dot(v,v), init=ones(3))  

# As in model 1, by providing additionally the gradient of the log of Normal(0, 1), which is v->-2v
mymodel2 = model(v-> -dot(v,v), grad=v->-2v, init=ones(3))   

### Second way of defining the model by using expression parsing and automatic differentiation

modelxpr = quote
    v ~ Normal(0, 1)
end

mymodel3 = model(modelxpr, v=ones(3)) # without specifying the gradient
mymodel4 = model(modelxpr, gradient=true, v=ones(3)) # with gradient
```

#### Single chain run

A single Monte Carlo chain can be simulated either by using the `run()` function or the `*` operator. For example, a
Random Walk Metropolis (RWM) sampler with scale parameter equal to 0.1 is run for `mymodel1`.

```jl
### Syntax 1 (using run())

# burnin = 100, keeping iterations 101 to 1000 (900 post-burnin iterations)
mychain = run(mymodel1 * RWM(0.1), steps=1000, burnin=100)

# burnin = 100, keeping every 5th iteration between 101 and 1000 (180 post-burnin iterations)
mychain = run(mymodel1 * RWM(0.1), steps=1000, burnin=100, thinning=5)

# As before, but expressed alternatively by using a range for steps
mychain = run(mymodel1 * RWM(0.1), steps=101:5:1000)

### Syntax 2 (using the `*` operator formatted as model * sampler * range)

mychain = mymodel1 * RWM(0.1) * (101:5:1000)  
```

#### Output summary and printing

The output of the MCMC simulation is stored in a composite type called `MCMCChain` whose `samples` and `gradients`
fields are dataframes containing the simulated chain and its gradient respectively. Being dataframes, these fields can
be printed, summarized and manipulated using the relevant facilities of the `DataFrames` package.

```jl
using DataFrames

# Print samples and diagnostics
head(mychain.samples)
head(mychain.diagnostics)

# Get a summary of the simulated chain
describe(mychain.samples)
```

#### Resuming MCMC simulation

To resume simulation whence it was left, the `run()` function is invoked on the chain.

```jl
mychain = run(mychain, steps=10000)
```

#### Model and sampler specifications must match

`mymodel3` does not provide the gradient of the log-target, since the `gradient` named argument of `model()` defaults
to `false`. Therefore, `mymodel3` can not be run in combination with a sampler that requires the gradient. On the other
hand, `mymodel4` provides the log-target's gradient that is needed by some MCMC routines such as HMC or MALA.

```jl
mymodel3 * MALA(0.1) * (1:1000) # Throws an error 

mymodel4 * MALA(0.1) * (1:1000) # It works
```

#### Running multiple chains

It is possible to run multiple independent chains for different samplers or for varying field values of the same
sampler.

```jl
# Run all 3 samplers via a single command
mychain = run(mymodel2 * [RWM(0.1), MALA(0.1), HMC(3,0.1)], steps=1000) 
mychain[2].samples  # prints samples for MALA(0.1)

# Run HMC with varying number of inner steps
mychain = run(mymodel2 * [HMC(i,0.1) for i in 1:5], steps=1000)
```

### Example 2: An example of a sequential Monte Carlo simulation

The following example demonstrates how to run a sequential Monte Carlo simulation.

```jl
using Distributions, MCMC

# Define a set of models that converge toward the distribution of interest (in the spirit of simulated annealing)
nmod = 10  # Number of models
mods = Array(MCMCLikModel, nmod)
sts = logspace(1, -1, nmod)
for i in 1:nmod
	m = quote
		y = abs(x)
		y ~ Normal(1, $(sts[i]))
	end
	mods[i] = model(m, x=0)
end

# Build MCMCTasks with diminishing scaling
targets = MCMCTask[mods[i] * RWM(sts[i]) for i in 1:nmod]

# Create 1000 particles
particles = [[randn()] for i in 1:1000]

# Launch sequential MC (10 steps x 1000 particles = 10000 samples returned in a single MCMCChain)
mychain1 = seqMC(targets, particles, steps=10, burnin=0)

# Resample with replacement using weights to approximate the real distribution
mychain2 = wsample(mychain1.samples["x"], mychain1.diagnostics["weigths"], 1000)

mean(mychain2)
```

The output of the sequential Monte Carlo simulation can be plotted with any graphical package. Below is an example using the `Vega` package.

```jl
using DataFrames, Vega

# Plot models
xx = linspace(-3,3,100) * ones(nmod)' 
yy = Float64[ mods[j].eval([xx[i,j]]) for i in 1:100, j in 1:nmod]
g = ones(100) * [1:10]'  
plot(x = vec(xx), y = exp(vec(yy)), group= vec(g), kind = :line)

# Plot a subset of raw samples
ts = collect(1:10:nrow(mychain1.samples))
plot(x = ts, y = vector(mychain1.samples[ts, "x"]), kind = :scatter)

# Plot weighted samples
plot(x = [1:1000], y = mychain2, kind = :scatter)
```

## Future Features

Future development is planned to provide:
- MCMC diagnostic tools and more extensive output analysis,
- Finer tuning and adaptive MCMC sampling,
- Extended automatic differentiation that covers MCMC samplers which use higher order derivatives of the log-target,
- Rejection, importance and slice sampling,
- Gibbs sampling and more generally sampling by assuming different target for each of the model parameters,
- Parallel implementation of population MCMC and its cluster job submission.
- Model DSL improvements such as loops, truncation / censoring 
