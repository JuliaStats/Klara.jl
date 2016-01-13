Lora.jl
==============================

[![Build Status](https://travis-ci.org/JuliaStats/Lora.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Lora.jl)
[![Lora](http://pkg.julialang.org/badges/Lora_0.3.svg)](http://pkg.julialang.org/?pkg=Lora&ver=0.3)
[![Lora](http://pkg.julialang.org/badges/Lora_0.4.svg)](http://pkg.julialang.org/?pkg=Lora&ver=0.4)
[![Docs](https://readthedocs.org/projects/lorajl/badge/?version=latest)](http://lorajl.readthedocs.org/en/latest/)
[![Stories in In Progress](https://badge.waffle.io/JuliaStats/Lora.jl.svg?label=In%20Progress&title=In%20Progress)](http://waffle.io/JuliaStats/Lora.jl)

[![Throughput Graph](https://graphs.waffle.io/JuliaStats/Lora.jl/throughput.svg)](https://waffle.io/JuliaStats/Lora.jl/metrics)

The Julia *Lora* package provides a generic engine for Markov Chain Monte Carlo (MCMC) inference.

*Lora* has undergone a major upgrade. Some of its recent changes include:

* Models are represented internally by graphs.
* Memory allocation and garbage collection have been reduced by using mutating functions associated with targets.
* It is possible to select storing output in memory or in file at runtime.
* Automatic differentiation is available allowing to choose between forward mode and reverse mode (the latter relying
on source transformation).

*To run the current version of Lora, it is needed to run `Pkg.checkout("Lora")`. Lora will be pushed to METADATA very soon.*

Some of the old one has not been fully ported. The full porting of old functionality, as well as further developments, will
be completed shortly. Progress is being tracked systematically via issues and milestones.

The documentation is out of date, but will be brought up-to-date fairly soon. In the meantime, this README file provides a
few examples of the new interface, explaining how to get up to speed with the new face of Lora.

Example: sampling from an unnormalized normal target
------------------------------

```julia
using Lora

### Define the log-target as a function (generic or anonymous):

plogtarget(z::Vector{Float64}) = -dot(z, z)

### Define the parameter via BasicContMuvParameter (it is a continuous multivariate variable)
### The input arguments for BasicContMuvParameter are:
### 1) the variable key,
### 2) the log-target

p = BasicContMuvParameter(:p, logtarget=plogtarget)

#### Define the model using the single_parameter_likelihood_model generator

model = single_parameter_likelihood_model(p, isindexed=false)

### Define a Metropolis-Hastings sampler with an identity covariance matrix

sampler = MH(ones(2))

### Set MCMC sampling range

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

### Set initial values for simulation

v0 = Dict(:p=>[5.1, -0.9])

### Specify job to be run

job = BasicMCJob(model, sampler, mcrange, v0)

### Run the simulation

run(job)

### Get simulated values

chain = output(job)

chain.value

### Check that the simulated values are close to the zero-mean target

[mean(chain.value[i, :]) for i in 1:2]
```

To reset the job, using a new initial value for the targeted parameter, run

```julia
reset(job, [3.2, 9.4])

run(job)

chain = output(job)
```

To see how the acceptance rate changes during burnin, set the vanilla tuner in verbose mode

```julia
job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true))

run(job)

chain = output(job)
```

If apart from the simulated chain you also want to store the log-target, then pass an additional dictionary to the job to
specify the output options. In particular, the `:monitor` key indicates which items will be monitored. In the example below,
both `:value` and `:logtarget` will be monitored, referring to the chain and log-target respectively. These can then be
accessed by the corresponding fields `chain.value` and `chain.logtarget`:

```julia
outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)

chain.logtarget
```

The acceptance ratio diagnostics can be stgored via the `:diagnostics=>[:accept]` entry of `outopts`:

```julia
outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)
```

Instead of saving the output in memory, it can be written in file via the output option `:destination=>:iostream`:

```julia
outopts = Dict{Symbol, Any}(
  :monitor=>[:value, :logtarget],
  :diagnostics=>[:accept],
  :destination=>:iostream
)

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)
```

The chain, log-target and acceptance ratio diagnostics of the above example are stored in the respective CSV files
"value.csv", "logtarget.csv" and "diagnosticvalues.csv" of the current directory. To save the output in another directory,
use the `:filepath=>"myfullpath.csv"`, where "myfullpath.csv" is substituted by the full path of your choice:

```julia
outopts = Dict{Symbol, Any}(
  :monitor=>[:value, :logtarget],
  :diagnostics=>[:accept],
  :destination=>:iostream,
  :filepath=>"/Users/theodore/workspace/julia"
)

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)
```

To use Julia tasks for running the job, set `plain=false`:

```julia
outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts, plain=false)

run(job)

chain = output(job)
```

Task-based jobs can also be reset:

```julia
reset(job, [-2.8, 3.4])

run(job)

chain = output(job)
```

To run a sampler which requires the gradient of the log-target, such as MALA, try

```julia
using Lora

plogtarget(z::Vector{Float64}) = -dot(z, z)

pgradlogtarget(z::Vector{Float64}) = -2*z

p = BasicContMuvParameter(:p, logtarget=plogtarget, gradlogtarget=pgradlogtarget)

model = single_parameter_likelihood_model(p)

### Set driftstep to 0.9

sampler = MALA(0.9)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>[5.1, -0.9])

### Save grad-log-target along with the chain (value and log-target)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)

chain.gradlogtarget

[mean(chain.value[i, :]) for i in 1:2]
```

To adapt the MALA drift step empirically during burnin towards an intended acceptance rate of 60%, run

```julia
job = BasicMCJob(
  model,
  sampler,
  mcrange,
  v0,
  tuner=AcceptanceRateMCTuner(0.6, verbose=true),
  outopts=outopts
)

run(job)

chain = output(job)
```

The examples below demonstrates how to run MCMC using automatic differentiation (AD).

To use forward mode AD, try the following:

```julia
using Lora

plogtarget(z::Vector) = -dot(z, z)

p = BasicContMuvParameter(:p, logtarget=plogtarget, autodiff=:forward)

model = single_parameter_likelihood_model(p)

sampler = MALA(0.9)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>[5.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)
```

Note that `plogtarget` takes an argument of type `Vector` instead of `Vector{Float64}`, as required by the ForwardDiff
package. Furthermore, notice that in the definition of parameter `p`, the gradient of its log-target is not provided
explicitly; instead, the optional argument `autodiff=:forward` enables computing the gradient via forward mode AD.

To employ reverse mode AD, try

```julia
using Lora

plogtarget(z::Vector) = -dot(z, z)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  autodiff=:reverse,
  init=[nothing, nothing, (ones(2),)]
)

model = single_parameter_likelihood_model(p)

sampler = MALA(0.9)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>[5.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)
```

In this case the optional argument `autodiff=:reverse` enables computing the gradient via reverse mode AD using
source transformation. Notice also the `init=[nothing, nothing, (ones(2),)]` optional argument, which allows passing the
required input to the `init` optional argument of `rdiff()` of the ReverseDiffSource package. From the Lora side,
`init` is a 3-element vector with the first, second and third arguments corresponding to the `init` arguments passed to
`rdiff()` for the log-likelihood, log-prior and log-target. If some of these functions is on specified, then the
respective `init` element is set to `nothing`.

Finally, it is possible to run reverse mode AD by passing an expression for the log-target (or log-likelihood or
log-prior) instead of a function. An example follows where the log-target is specified via an expression:

```julia
using Lora

p = BasicContMuvParameter(
  :p,
  logtarget=:(-dot(z, z)),
  autodiff=:reverse,
  init=[nothing, nothing, (:z, ones(2))]
)

model = single_parameter_likelihood_model(p)

sampler = MALA(0.9)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>[5.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)
```

Documentation
------------------------------

Please note that the documentation needs to be updated and it does not reflect yet the current version of *Lora*.

* [User Guide](http://lorajl.readthedocs.org/en/latest/) ([PDF](https://readthedocs.org/projects/lorajl/downloads/pdf/latest/))
* Cheat Sheet (to appear soon)
