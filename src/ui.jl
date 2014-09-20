function run(model::MCModel, sampler::MCSampler, runner::SerialMC, tuner::MCTuner=VanillaMCTuner(), job::Symbol=:plain)
  if job == :plain
    run(model, sampler, runner, tuner, PlainMCJob(model, sampler, runner, tuner))
  elseif job == :task
    run(model, sampler, runner, tuner, TaskMCJob(model, sampler, runner, tuner))
  end
end

### The MCSystem is not useful only as an alternative interface
### It gathers all the low and high level components that fully specify a Monte Carlo simulation
### A particular instance which demonstrates the usefulness of MCSystem is the case of storing the Monte Carlo task
### Storing the task of a task-based job in MCSystem.job.task allows resuming a Monte Carlo simulation

function MCSystem(model::MCModel, sampler::MCSampler, runner::MCRunner, tuner::MCTuner=VanillaMCTuner(),
  job::Symbol=:plain)
  if job == :plain
    MCSystem(model, sampler, runner, tuner, PlainMCJob(model, sampler, runner, tuner))
  elseif job == :task
    MCSystem(model, sampler, runner, tuner, TaskMCJob(model, sampler, runner, tuner))
  end
end

run(system::MCSystem) = run(system.model, system.sampler, system.runner, system.tuner, system.job)

### A third form of interface is via the function wrappers provided below
### This third interface fulfills the following purposes:
### i) It acts as a simplified interface to facilitate interaction with other packages
### ii) It provides a shorter syntax by creating the unerlying model, sampler and runner types under the hook
### iii) It fits better the mindset of some users

function HMC(f::Function, g::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, nleaps::Int=10, leapstep::Float64=0.1, thinning::Int=1)
  mcmodel::MCLikModel = model(f, grad=g, init=init)
  mcsampler::HMC = HMC(nleaps, leapstep)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end
