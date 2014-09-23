function run(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner=VanillaMCTuner(), job::Symbol=:plain)
  if job == :plain
    run(m, s, r, t, PlainMCJob(m, s, r, t))
  elseif job == :task
    run(m, s, r, t, TaskMCJob(m, s, r, t))
  end
end

### The MCSystem is not useful only as an alternative interface
### It gathers all the low and high level components that fully specify a Monte Carlo simulation
### A particular instance which demonstrates the usefulness of MCSystem is the case of storing the Monte Carlo task
### Storing the task of a task-based job in MCSystem.job.task allows resuming a Monte Carlo simulation

function MCSystem(m::MCModel, s::MCSampler, r::MCRunner, t::MCTuner=VanillaMCTuner(), job::Symbol=:plain)
  if job == :plain
    MCSystem(m, s, r, t, PlainMCJob(m, s, r, t))
  elseif job == :task
    MCSystem(m, s, r, t, TaskMCJob(m, s, r, t))
  end
end

run(s::MCSystem) = run(s.model, s.sampler, s.runner, s.tuner, s.job)

### A third form of interface is via the function wrappers provided below
### This third interface fulfills the following purposes:
### i) It acts as a simplified interface to facilitate interaction with other packages
### ii) It provides a shorter syntax by creating the unerlying model, sampler and runner types under the hook
### iii) It fits better the mindset of some users

function MH(f::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1,
  thinning::Int=1,
  logproposal::FunctionOrNothing=nothing,
  randproposal::FunctionOrNothing=(x::Vector{Float64} -> rand(IsoNormal(x, 1.))))
  mcmodel::MCLikModel = model(f, init=init)
  mcsampler::MH
  if logproposal==nothing
    mcsampler = MH(randproposal)
  else
    mcsampler = MH(logproposal, randproposal)
  end
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function HMC(f::Function, g::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, thinning::Int=1, nleaps::Int=10, leapstep::Float64=0.1)
  mcmodel::MCLikModel = model(f, grad=g, init=init)
  mcsampler::HMC = HMC(nleaps, leapstep)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function MALA(f::Function, g::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, thinning::Int=1, driftstep::Float64=1.)
  mcmodel::MCLikModel = model(f, grad=g, init=init)
  mcsampler::MALA = MALA(driftstep=driftstep)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function SMMALA(f::Function, g::Function, h::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, thinning::Int=1, driftstep::Float64=1.)
  mcmodel::MCLikModel = model(f, grad=g, tensor=h, init=init)
  mcsampler::SMMALA = SMMALA(driftstep=driftstep)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end
