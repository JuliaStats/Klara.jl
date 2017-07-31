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
