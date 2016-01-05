using Lora

### Define the vector of keys referring to model variables
### For this naive example, the model consists of a single parameter represented by the :p key

vkeys = [:p]

### Define the log-target as a function (generic or anonymous):

plogtarget(z::Vector{Float64}) = -dot(z, z)

### Define the parameter via BasicContMuvParameter (it is a continuous multivariate variable)
### The input arguments for BasicContMuvParameter are:
### 1) vkeys, the vector of variable keys,
### 2) the index of the current variable in vkeys,
### 3) the log-target

p = BasicContMuvParameter(vkeys, 1, logtarget=plogtarget, nkeys=0)

#### Define the model using the single_parameter_likelihood_model generator

model = single_parameter_likelihood_model(p)

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

reset(job, [3.2, 9.4])

run(job)

chain = output(job)

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true))

run(job)

chain = output(job)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)

chain.logtarget

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

filepath = dirname(@__FILE__)
filesuffix = "csv"

outopts = Dict{Symbol, Any}(
  :monitor=>[:value, :logtarget],
  :diagnostics=>[:accept],
  :destination=>:iostream,
  :filepath=>filepath
)

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

for fname in (:value, :logtarget, :diagnosticvalues)
  rm(joinpath(filepath, string(fname)*"."*filesuffix))
end

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts, plain=false)

run(job)

chain = output(job)

reset(job, [-2.8, 3.4])

run(job)

chain = output(job)

vkeys = [:p]

plogtarget(z::Vector{Float64}) = -dot(z, z)

pgradlogtarget(z::Vector{Float64}) = -2*z

p = BasicContMuvParameter(vkeys, 1, logtarget=plogtarget, gradlogtarget=pgradlogtarget, nkeys=0)

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

vkeys = [:p]

plogtarget(z::Vector) = -dot(z, z)

p = BasicContMuvParameter(vkeys, 1, logtarget=plogtarget, autodiff=:forward, nkeys=0)

model = single_parameter_likelihood_model(p)

sampler = MALA(0.9)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>[5.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)

vkeys = [:p]

plogtarget(z::Vector) = -dot(z, z)

p = BasicContMuvParameter(vkeys, 1, logtarget=plogtarget, autodiff=:reverse, nkeys=0, init=[nothing, nothing, (ones(2),)])

model = single_parameter_likelihood_model(p)

sampler = MALA(0.9)

mcrange = BasicMCRange(nsteps=10000, burnin=1000)

v0 = Dict(:p=>[5.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=true), outopts=outopts)

run(job)

chain = output(job)

vkeys = [:p]

p = BasicContMuvParameter(
  vkeys,
  1,
  logtarget=:(-dot(z, z)),
  autodiff=:reverse,
  nkeys=0,
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
