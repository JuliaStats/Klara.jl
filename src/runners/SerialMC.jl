###########################################################################
#
#  SerialMC runner: consumes repeatedly a sampler and returns a MCMCChain
#
#
###########################################################################

export SerialMC

println("Loading SerialMC(steps, burnin, thinning) runner")

immutable SerialMC <: MCMCRunner
  burnin::Int
  thinning::Int
  len::Int
  r::Range

  function SerialMC(steps::Range{Int})
    r = steps

    burnin = first(r)-1
    thinning = r.step
    len = last(r)

    @assert burnin >= 0 "Burnin rounds ($burnin) should be >= 0"
    @assert len > burnin "Total MCMC length ($len) should be > to burnin ($burnin)"
    @assert thinning >= 1 "Thinning ($thinning) should be >= 1"

    new(burnin, thinning, len, r)
  end
end

SerialMC(steps::Range1{Int}) = SerialMC(first(steps):1:last(steps))

SerialMC(; steps::Int=100, burnin::Int=0, thinning::Int=1) = SerialMC((burnin+1):thinning:steps)

function run_serialmc(t::MCMCTask)
  tic() # start timer

  # array allocations to store results
  samples        = fill(NaN, t.model.size, length(t.runner.r))
  gradients      = fill(NaN, t.model.size, length(t.runner.r))
  diags          = {"step" => collect(t.runner.r)}
  logtargets     = fill(NaN, length(t.runner.r))

  # sampling loop
  j = 1
  for i in 1:t.runner.len
    newprop = consume(t.task)
    if in(i, t.runner.r)
      samples[:, j] = newprop.ppars
      logtargets[j] = newprop.plogtarget

      if newprop.pgrads != nothing
        gradients[:, j] = newprop.pgrads
      end

      # save diagnostics
      for (k,v) in newprop.diagnostics
        # if diag name not seen before, create column
        if !haskey(diags, k)
          diags[k] = Array(typeof(v), length(diags["step"]))          
        end
        
        diags[k][j] = v
      end

      j += 1
    end
  end

  # create Chain
  MCMCChain(t.runner.r, samples', logtargets, gradients', diags, t, toq())
end

function run_serialmc_exit(t::MCMCTask)
  chain = run_serialmc(t)
  stop!(chain)
  return chain
end

function resume_serialmc(t::MCMCTask; steps::Int=100)
  @assert typeof(t.runner) == SerialMC
    "resume_serialmc can not be called on an MCMCTask whose runner is of type $(fieldtype(t, :runner))"
  run(t.model, t.sampler, SerialMC(steps=steps, thinning=t.runner.thinning))
end
