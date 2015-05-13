### THIS IS WIP, runner under construction

### Serial tempering Monte Carlo runner

export SerialTempMC

immutable SerialTemperingMCRunner <: SerialMCRunner
  burnin::Int
  thinning::Int
  nsteps::Int
  r::Range{Int}
  swapperiod::Int

  function SerialTemperingMCRunner(r::Range{Int}, p::Int)
    burnin = first(r)-1
    thinning = r.step
    nsteps = last(r)
    @assert burnin >= 0 "Number of burn-in iterations should be non-negative."
    @assert thinning >= 1 "Thinning should be >= 1."
    @assert nsteps > burnin "Total number of MCMC iterations should be greater than number of burn-in iterations."
    @assert p > 0 "Swap period should be positive."
    new(burnin, thinning, nsteps, r, p)
  end
end

SerialTemperingMCRunner(r::UnitRange{Int}, p::Int) = SerialTemperingMCRunner(first(r):1:last(r), p)

SerialTemperingMCRunner(; burnin::Int=0, thinning::Int=1, nsteps::Int=100, swapperiod::Int=5) =
  SerialTemperingMCRunner((burnin+1):thinning:nsteps, swapperiod)

typealias SerialTempMC SerialTemperingMCRunner

function run(systems::Vector{MCSystem})
  tic()

  local nchains = length(systems)
  local modelsize = systems[1].model.size
  local runner = systems[1].runner
  local jobtype = typeof(systems[1].job)

  @assert all(map(s->s.model.size, systems) .== modelsize) "Models must have the same parameter vector size."
  @assert isa(runner, SerialTempMC) "Runners must be of SerialTempMC type."
  @assert all(map(s->s.runner, systems) .== runner) "The runners in the vector of MC systems must be identical."
  @assert all(map(s->s.job, systems) .== jobtype) "Jobs must be of the same type."

	local logweights = zeros(nchains)  # log of task weights that will be adapted
  local at1 = 1  # Pick starting task

  # Pre-allocation for storing results
  mcchain::MCChain = MCChain(modelsize, length(runner.r))

  local mcstate1 = systems[at1].job.receive()
  local ppars, logtarget, pars
  ppars, logtarget, pars = mcstate1.ppars, mcstate1.logtarget, mcstate1.pars

  # Sampling loop
  i::Int = 1
  for j in 1:runner.nsteps
    if j % swapperiod == 0  # Attempt a task switch
      at2 = rand(1:(nchains-1)) # Pick another task at1 random
      at2 = at2 >= at1 ? at2+1 : at2

      mcstate2 = systems[at2].job.reset(pars)
      if rand() < exp(logtarget - mcstate2.logtarget + logweights[at2] - logweights[at1]) # accept swap ?
        at1, mcstate1 = at2, mcstate2
      end
    else
      mcstate1 = consume(systems[at1].task)
    end

    # TODO : add logweights adaptation
    ppars, logtarget, pars = mcstate1.ppars, mcstate1.logtarget, mcstate1.pars

    if in(j, runner.r)
      mcchain.samples[i, :] = ppars
      mcchain.logtargets[i] = logtarget

      i += 1
      #mcchain.misc[:mod][pos] = at1
    end
  end

  mcchain.runtime = toq()
  mcchain
end

# TODO: check that all elements of array contain MCMCTasks of the same type
function resume(systems::Array{MCMCTask}; nsteps::Int=100)
  run(MCMCTask[systems[j].model * systems[j].sampler * SerialTempMC(nsteps=nsteps, swapperiod=systems[j].runner.swapperiod)
    for j in 1:length(systems)])
end
