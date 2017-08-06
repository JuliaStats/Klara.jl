Klara.jl
==============================

[![Build Status](https://travis-ci.org/JuliaStats/Klara.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Klara.jl)
[![Klara](http://pkg.julialang.org/badges/Klara_0.6.svg)](http://pkg.julialang.org/?pkg=Klara&ver=0.6)
[![Docs](https://readthedocs.org/projects/klarajl/badge/?version=latest)](http://klarajl.readthedocs.org/en/latest/)
[![Stories in In Progress](https://badge.waffle.io/JuliaStats/Klara.jl.svg?label=In%20Progress&title=In%20Progress)](http://waffle.io/JuliaStats/Klara.jl)

[![Throughput Graph](https://graphs.waffle.io/JuliaStats/Klara.jl/throughput.svg)](https://waffle.io/JuliaStats/Klara.jl/metrics)

The Julia *Klara* package provides a generic engine for Markov Chain Monte Carlo (MCMC) inference.

The documentation is out of date, but will be brought up-to-date upon completing milestone 0.12.0. In the meantime, this README file provides some elementary examples of the current interface. More examples can be found in doc/examples.

Example: sampling from an unnormalized normal target
------------------------------

```julia
using Klara

### Define the log-target as a function (generic or anonymous):

plogtarget(z::Vector{Float64}) = -dot(z, z)

### Define the parameter via BasicContMuvParameter (it is a continuous multivariate variable)
### The input arguments for BasicContMuvParameter are:
### 1) the variable key,
### 2) the log-target

p = BasicContMuvParameter(:p, logtarget=plogtarget)

### Define the model using the likelihood_model generator
### The second argument informs the likelihood_model generator that p.index has not been set

model = likelihood_model(p, false)

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

mean(chain)
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

To run a sampler which requires the gradient of the log-target, such as MALA, try

```julia
using Klara

plogtarget(z::Vector{Float64}) = -dot(z, z)

pgradlogtarget(z::Vector{Float64}) = -2*z

p = BasicContMuvParameter(:p, logtarget=plogtarget, gradlogtarget=pgradlogtarget)

model = likelihood_model(p, false)

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

mean(chain)
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
using Klara

plogtarget(z::Vector) = -dot(z, z)

p = BasicContMuvParameter(:p, logtarget=plogtarget, diffopts=DiffOptions(mode=:forward))

model = likelihood_model(p, false)

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
explicitly; instead, the optional argument `diffopts=DiffOptions(mode=:forward)` enables computing the gradient via forward
mode AD.

To employ reverse mode AD, try

```julia
using Klara

plogtarget(z) = -dot(z, z)

p = BasicContMuvParameter(:p, logtarget=plogtarget, diffopts=DiffOptions(mode=:reverse))

model = likelihood_model(p, false)

sampler = MALA(0.9)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>[5.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)
```

`plogtarget` takes an argument without specifying its type, as required by the ReverseDiff package. In the definition of
parameter `p`, the gradient of its log-target is not provided explicitly; the optional argument
`diffopts=DiffOptions(mode=:reverse)` enables computing the gradient via reverse mode AD.

Documentation
------------------------------

The user guide is currently being written up.

* [User Guide](http://klarajl.readthedocs.org/en/latest/) ([PDF](https://readthedocs.org/projects/klarajl/downloads/pdf/latest/))
