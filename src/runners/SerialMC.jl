### Serial "ordinary" Monte Carlo runner

immutable SerialMCBaseRunner <: SerialMCRunner
  burnin::Int
  thinning::Int
  nsteps::Int
  r::Range{Int}
  storegradlogtarget::Bool # Indicates whether to save the gradient of the log-target in the cases it is available

  function SerialMCBaseRunner(r::Range{Int}, s::Bool=false)
    burnin = first(r)-1
    thinning = r.step
    nsteps = last(r)
    @assert burnin >= 0 "Number of burn-in iterations should be non-negative."
    @assert thinning >= 1 "Thinning should be >= 1."
    @assert nsteps > burnin "Total number of MCMC iterations should be greater than number of burn-in iterations."
    new(burnin, thinning, nsteps, r, s)
  end
end

SerialMCBaseRunner(r::Range1{Int}, s::Bool=false) = SerialMCBaseRunner(first(r):1:last(r), s)

SerialMCBaseRunner(; burnin::Int=0, thinning::Int=1, nsteps::Int=100, storegradlogtarget::Bool=false) =
  SerialMCBaseRunner((burnin+1):thinning:nsteps, storegradlogtarget)

typealias SerialMC SerialMCBaseRunner

function run(m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner=VanillaMCTuner(), job::Symbol=:task)
  tic()

  mcjob = MCJob(m, s, r, t, job)

  # Pre-allocation for storing results
  mcchain::MCChain = MCChain(m.size, length(r.r); storegradlogtarget=r.storegradlogtarget)
  ds = Dict{Any, Any}("step" => collect(r.r))

  # Sampling loop
  i::Int = 1
  for j in 1:r.nsteps
    mcstate = mcjob.receive(1)
    if in(j, r.r)
      mcchain.samples[i, :] = mcstate.successive.sample
      mcchain.logtargets[i] = mcstate.successive.logtarget

      if r.storegradlogtarget
        mcchain.gradlogtargets[i, :] = mcstate.successive.gradlogtarget
      end

      # Save diagnostics
      for (k,v) in mcstate.diagnostics
        # If diagnostics name not seen before, create column
        if !haskey(ds, k)
          ds[k] = Array(typeof(v), length(ds["step"]))          
        end
        
        ds[k][i] = v
      end

      i += 1
    end
  end

  mcchain.diagnostics, mcchain.runtime = ds, toq()
  mcchain
end

function resume!(c::MCChain, m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner=VanillaMCTuner(), j::Symbol=:task;
  nsteps::Int=100)
  m.init = vec(c.samples[end, :])
  mcrunner::SerialMC = SerialMC(burnin=0, thinning=r.thinning, nsteps=nsteps, storegradlogtarget=r.storegradlogtarget)
  run(m, s, mcrunner, t, j)
end

resume(c::MCChain, m::MCModel, s::MCSampler, r::SerialMC, t::MCTuner=VanillaMCTuner(), j::Symbol=:task;
  nsteps::Int=100) =
  resume!(c, deepcopy(m), s, r, t, j; nsteps=nsteps)
