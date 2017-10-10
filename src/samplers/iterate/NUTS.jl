function iterate!(job::BasicMCJob, ::Type{NUTS}, ::Type{Univariate})
  local a::Real
  local na::Integer

  if isa(job.tuner, DualAveragingMCTuner)
    job.sstate.count += 1
  end

  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.momentum = randn()

  job.sstate.oldhamiltonian = hamiltonian(job.pstate.logtarget, job.sstate.momentum)

  job.sstate.pstateplus.value = job.pstate.value
  job.sstate.pstateplus.gradlogtarget = job.pstate.gradlogtarget
  job.sstate.momentumplus = job.sstate.momentum
  job.sstate.pstateminus.value = job.pstate.value
  job.sstate.pstateminus.gradlogtarget = job.pstate.gradlogtarget
  job.sstate.momentumminus = job.sstate.momentum

  job.sstate.j = 0
  job.sstate.n = 1
  job.sstate.s = true
  if in(:accept, job.outopts[:diagnostics]) || job.tuner.verbose
    job.sstate.update = false
  end

  job.sstate.u = log(rand())+job.sstate.oldhamiltonian

  while (job.sstate.s && (job.sstate.j < job.sampler.maxndoublings))
    job.sstate.v = rand(Bool) ? 1 : -1

    if isa(job.tuner, VanillaMCTuner)
      if job.sstate.v == -1
        job.sstate.pstateminus,
        job.sstate.momentumminus,
        _,
        _,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime = build_tree!(
          job.sstate,
          job.sstate.pstateminus,
          job.sstate.momentumminus,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          VanillaMCTuner,
          job.outopts
        )
      else
        _,
        _,
        job.sstate.pstateplus,
        job.sstate.momentumplus,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime = build_tree!(
          job.sstate,
          job.sstate.pstateplus,
          job.sstate.momentumplus,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          VanillaMCTuner,
          job.outopts
        )
      end
    elseif isa(job.tuner, DualAveragingMCTuner)
      if job.sstate.v == -1
        job.sstate.pstateminus,
        job.sstate.momentumminus,
        _,
        _,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime,
        a,
        na = build_tree!(
          job.sstate,
          job.sstate.pstateminus,
          job.sstate.momentumminus,
          job.sstate.oldhamiltonian,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          DualAveragingMCTuner,
          job.outopts
        )
      else
        _,
        _,
        job.sstate.pstateplus,
        job.sstate.momentumplus,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime,
        a,
        na = build_tree!(
          job.sstate,
          job.sstate.pstateplus,
          job.sstate.momentumplus,
          job.sstate.oldhamiltonian,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          DualAveragingMCTuner,
          job.outopts
        )
      end
    end

    if (job.sstate.sprime && (rand() < job.sstate.nprime/job.sstate.n))
      job.pstate.value = job.sstate.pstateprime.value

      job.pstate.gradlogtarget = job.sstate.pstateprime.gradlogtarget
      if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
        job.pstate.gradloglikelihood = job.sstate.pstateprime.gradloglikelihood
      end
      if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
        job.pstate.gradlogprior = job.sstate.pstateprime.gradlogprior
      end

      job.pstate.logtarget = job.sstate.pstateprime.logtarget
      if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
        job.pstate.loglikelihood = job.sstate.pstateprime.loglikelihood
      end
      if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
        job.pstate.logprior = job.sstate.pstateprime.logprior
      end

      if in(:accept, job.outopts[:diagnostics]) || job.tuner.verbose
        job.sstate.update = true
      end
    end

    job.sstate.j += 1
    job.sstate.n += job.sstate.nprime
    job.sstate.s =
      job.sstate.sprime &&
      !uturn(job.sstate.pstateplus.value, job.sstate.pstateminus.value, job.sstate.momentumplus, job.sstate.momentumminus)
  end

  if !isempty(job.sstate.diagnosticindices)
    if haskey(job.sstate.diagnosticindices, :accept)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = job.sstate.update
    end

    if haskey(job.sstate.diagnosticindices, :ndoublings)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:ndoublings]] = job.sstate.j
    end

    if haskey(job.sstate.diagnosticindices, :a)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:a]] = a
    end

    if haskey(job.sstate.diagnosticindices, :na)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:na]] = na
    end
  end

  if job.tuner.verbose
    job.sstate.update && (job.sstate.tune.accepted += 1)
  end

  if isa(job.tuner, VanillaMCTuner) && job.tuner.verbose
    if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
      rate!(job.sstate.tune)

      if job.tuner.verbose
        println(
          "Burnin iteration ",
          job.fmt_iter(job.sstate.tune.totproposed),
          " of ",
          job.range.burnin,
          ": ",
          job.fmt_perc(100*job.sstate.tune.rate),
          " % acceptance rate"
        )
      end

      reset_burnin!(job.sstate.tune)
    end
  elseif isa(job.tuner, DualAveragingMCTuner)
    if (job.sstate.count <= job.tuner.nadapt)
      tune!(job.sstate.tune, job.tuner, job.sstate.count, a/na)
    else
      job.sstate.tune.step = job.sstate.tune.εbar
    end

    if job.tuner.verbose
      if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
        rate!(job.sstate.tune)

        println(
          "Burnin iteration ",
          job.fmt_iter(job.sstate.tune.totproposed),
          " of ",
          job.range.burnin,
          ": ",
          job.fmt_perc(100*job.sstate.tune.rate),
          " % acceptance rate"
        )

        reset_burnin!(job.sstate.tune)
      end
    end
  end
end

function iterate!(job::BasicMCJob, ::Type{NUTS}, ::Type{Multivariate})
  local a::Real
  local na::Integer

  if isa(job.tuner, DualAveragingMCTuner)
    job.sstate.count += 1
  end

  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.momentum[:] = randn(job.pstate.size)

  job.sstate.oldhamiltonian = hamiltonian(job.pstate.logtarget, job.sstate.momentum)

  job.sstate.pstateplus.value = copy(job.pstate.value)
  job.sstate.pstateplus.gradlogtarget = copy(job.pstate.gradlogtarget)
  job.sstate.momentumplus = copy(job.sstate.momentum)
  job.sstate.pstateminus.value = copy(job.pstate.value)
  job.sstate.pstateminus.gradlogtarget = copy(job.pstate.gradlogtarget)
  job.sstate.momentumminus = copy(job.sstate.momentum)

  job.sstate.j = 0
  job.sstate.n = 1
  job.sstate.s = true
  if in(:accept, job.outopts[:diagnostics]) || job.tuner.verbose
    job.sstate.update = false
  end

  job.sstate.u = log(rand())+job.sstate.oldhamiltonian

  while (job.sstate.s && (job.sstate.j < job.sampler.maxndoublings))
    job.sstate.v = rand(Bool) ? 1 : -1

    if isa(job.tuner, VanillaMCTuner)
      if job.sstate.v == -1
        job.sstate.pstateminus,
        job.sstate.momentumminus,
        _,
        _,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime = build_tree!(
          job.sstate,
          job.sstate.pstateminus,
          job.sstate.momentumminus,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          VanillaMCTuner,
          job.outopts
        )
      else
        _,
        _,
        job.sstate.pstateplus,
        job.sstate.momentumplus,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime = build_tree!(
          job.sstate,
          job.sstate.pstateplus,
          job.sstate.momentumplus,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          VanillaMCTuner,
          job.outopts
        )
      end
    elseif isa(job.tuner, DualAveragingMCTuner)
      if job.sstate.v == -1
        job.sstate.pstateminus,
        job.sstate.momentumminus,
        _,
        _,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime,
        a,
        na = build_tree!(
          job.sstate,
          job.sstate.pstateminus,
          job.sstate.momentumminus,
          job.sstate.oldhamiltonian,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          DualAveragingMCTuner,
          job.outopts
        )
      else
        _,
        _,
        job.sstate.pstateplus,
        job.sstate.momentumplus,
        job.sstate.pstateprime,
        job.sstate.nprime,
        job.sstate.sprime,
        a,
        na = build_tree!(
          job.sstate,
          job.sstate.pstateplus,
          job.sstate.momentumplus,
          job.sstate.oldhamiltonian,
          job.sstate.u,
          job.sstate.v,
          job.sstate.j,
          job.sstate.tune.step,
          job.parameter,
          job.sampler.maxδ,
          NUTS,
          DualAveragingMCTuner,
          job.outopts
        )
      end
    end

    if (job.sstate.sprime && (rand() < job.sstate.nprime/job.sstate.n))
      job.pstate.value = copy(job.sstate.pstateprime.value)

      job.pstate.gradlogtarget = copy(job.sstate.pstateprime.gradlogtarget)
      if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
        job.pstate.gradloglikelihood = copy(job.sstate.pstateprime.gradloglikelihood)
      end
      if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
        job.pstate.gradlogprior = copy(job.sstate.pstateprime.gradlogprior)
      end

      job.pstate.logtarget = job.sstate.pstateprime.logtarget
      if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
        job.pstate.loglikelihood = job.sstate.pstateprime.loglikelihood
      end
      if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
        job.pstate.logprior = job.sstate.pstateprime.logprior
      end

      if in(:accept, job.outopts[:diagnostics]) || job.tuner.verbose
        job.sstate.update = true
      end
    end

    job.sstate.j += 1
    job.sstate.n += job.sstate.nprime
    job.sstate.s =
      job.sstate.sprime &&
      !uturn(job.sstate.pstateplus.value, job.sstate.pstateminus.value, job.sstate.momentumplus, job.sstate.momentumminus)
  end

  if !isempty(job.sstate.diagnosticindices)
    if haskey(job.sstate.diagnosticindices, :accept)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = job.sstate.update
    end

    if haskey(job.sstate.diagnosticindices, :ndoublings)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:ndoublings]] = job.sstate.j
    end

    if haskey(job.sstate.diagnosticindices, :a)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:a]] = a
    end

    if haskey(job.sstate.diagnosticindices, :na)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:na]] = na
    end
  end

  if job.tuner.verbose
    job.sstate.update && (job.sstate.tune.accepted += 1)
  end

  if isa(job.tuner, VanillaMCTuner) && job.tuner.verbose
    if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
      rate!(job.sstate.tune)

      if job.tuner.verbose
        println(
          "Burnin iteration ",
          job.fmt_iter(job.sstate.tune.totproposed),
          " of ",
          job.range.burnin,
          ": ",
          job.fmt_perc(100*job.sstate.tune.rate),
          " % acceptance rate"
        )
      end

      reset_burnin!(job.sstate.tune)
    end
  elseif isa(job.tuner, DualAveragingMCTuner)
    if (job.sstate.count <= job.tuner.nadapt)
      tune!(job.sstate.tune, job.tuner, job.sstate.count, a/na)
    else
      job.sstate.tune.step = job.sstate.tune.εbar
    end

    if job.tuner.verbose
      if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
        rate!(job.sstate.tune)

        println(
          "Burnin iteration ",
          job.fmt_iter(job.sstate.tune.totproposed),
          " of ",
          job.range.burnin,
          ": ",
          job.fmt_perc(100*job.sstate.tune.rate),
          " % acceptance rate"
        )

        reset_burnin!(job.sstate.tune)
      end
    end
  end
end
