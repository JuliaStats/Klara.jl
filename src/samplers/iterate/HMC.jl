function iterate!(job::BasicMCJob, ::Type{HMC}, ::Type{Univariate})
  if isa(job.tuner, DualAveragingMCTuner)
    job.sstate.count += 1
  end

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) ||
    isa(job.tuner, AcceptanceRateMCTuner) ||
    (isa(job.tuner, DualAveragingMCTuner) && job.tuner.verbose)
    job.sstate.tune.proposed += 1
  end

  job.sstate.momentum = randn()

  job.sstate.oldhamiltonian = hamiltonian(job.pstate.logtarget, job.sstate.momentum)

  job.sstate.pstate.value = job.pstate.value
  job.sstate.pstate.gradlogtarget = job.pstate.gradlogtarget

  if isa(job.tuner, DualAveragingMCTuner)
    job.sstate.nleaps = max(1, Int(round(job.sstate.tune.λ/job.sstate.tune.step)))
  end

  for i in 1:job.sstate.nleaps
    job.sstate.momentum = leapfrog!(
      job.sstate.pstate, job.sstate.pstate, job.sstate.momentum, job.sstate.tune.step, job.parameter.gradlogtarget!
    )
  end

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.newhamiltonian = hamiltonian(job.sstate.pstate.logtarget, job.sstate.momentum)

  job.sstate.ratio = job.sstate.newhamiltonian-job.sstate.oldhamiltonian

  job.sstate.a = min(1., exp(job.sstate.ratio))

  if rand() < job.sstate.a
    job.pstate.value = job.sstate.pstate.value

    job.pstate.gradlogtarget = job.sstate.pstate.gradlogtarget
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      job.pstate.gradloglikelihood = job.sstate.pstate.gradloglikelihood
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      job.pstate.gradlogprior = job.sstate.pstate.gradlogprior
    end

    job.pstate.logtarget = job.sstate.pstate.logtarget
    if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
      job.pstate.loglikelihood = job.sstate.pstate.loglikelihood
    end
    if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
      job.pstate.logprior = job.sstate.pstate.logprior
    end

    if !isempty(job.sstate.diagnosticindices)
      if haskey(job.sstate.diagnosticindices, :accept)
        job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = true
      end
    end

    if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) ||
      isa(job.tuner, AcceptanceRateMCTuner) ||
      (isa(job.tuner, DualAveragingMCTuner) && job.tuner.verbose)
      job.sstate.tune.accepted += 1
    end
  else
    if !isempty(job.sstate.diagnosticindices)
      if haskey(job.sstate.diagnosticindices, :accept)
        job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
      end
    end
  end

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    if job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0
      rate!(job.sstate.tune)

      if isa(job.tuner, AcceptanceRateMCTuner)
        tune!(job.sstate.tune, job.tuner)
      end

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
    if job.sstate.count <= job.tuner.nadapt
      tune!(job.sstate.tune, job.tuner, job.sstate.count, job.sstate.a)

      if job.tuner.verbose
        if mod(job.sstate.tune.proposed, job.tuner.period) == 0
          rate!(job.sstate.tune)

          println(
            "Burnin iteration ",
            job.fmt_iter(job.sstate.tune.totproposed),
            " of ",
            job.tuner.nadapt,
            ": ",
            job.fmt_perc(100*job.sstate.tune.rate),
            " % acceptance rate"
          )

          reset_burnin!(job.sstate.tune)
        end
      end
    else
      job.sstate.tune.step = job.sstate.tune.εbar
    end
  end
end

function iterate!(job::BasicMCJob, ::Type{HMC}, ::Type{Multivariate})
  if isa(job.tuner, DualAveragingMCTuner)
    job.sstate.count += 1
  end

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) ||
    isa(job.tuner, AcceptanceRateMCTuner) ||
    (isa(job.tuner, DualAveragingMCTuner) && job.tuner.verbose)
    job.sstate.tune.proposed += 1
  end

  job.sstate.momentum[:] = randn(job.pstate.size)

  job.sstate.oldhamiltonian = hamiltonian(job.pstate.logtarget, job.sstate.momentum)

  job.sstate.pstate.value = copy(job.pstate.value)
  job.sstate.pstate.gradlogtarget = copy(job.pstate.gradlogtarget)

  if isa(job.tuner, DualAveragingMCTuner)
    job.sstate.nleaps = max(1, Int(round(job.sstate.tune.λ/job.sstate.tune.step)))
  end

  for i in 1:job.sstate.nleaps
    leapfrog!(
      job.sstate.pstate,
      job.sstate.momentum,
      job.sstate.pstate,
      job.sstate.momentum,
      job.sstate.tune.step,
      job.parameter.gradlogtarget!
    )
  end

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.newhamiltonian = hamiltonian(job.sstate.pstate.logtarget, job.sstate.momentum)

  job.sstate.ratio = job.sstate.newhamiltonian-job.sstate.oldhamiltonian

  job.sstate.a = min(1., exp(job.sstate.ratio))

  if rand() < job.sstate.a
    job.pstate.value = copy(job.sstate.pstate.value)

    job.pstate.gradlogtarget = copy(job.sstate.pstate.gradlogtarget)
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      job.pstate.gradloglikelihood = copy(job.sstate.pstate.gradloglikelihood)
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      job.pstate.gradlogprior = copy(job.sstate.pstate.gradlogprior)
    end

    job.pstate.logtarget = job.sstate.pstate.logtarget
    if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
      job.pstate.loglikelihood = job.sstate.pstate.loglikelihood
    end
    if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
      job.pstate.logprior = job.sstate.pstate.logprior
    end

    if !isempty(job.sstate.diagnosticindices)
      if haskey(job.sstate.diagnosticindices, :accept)
        job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = true
      end
    end

    if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) ||
      isa(job.tuner, AcceptanceRateMCTuner) ||
      (isa(job.tuner, DualAveragingMCTuner) && job.tuner.verbose)
      job.sstate.tune.accepted += 1
    end
  else
    if !isempty(job.sstate.diagnosticindices)
      if haskey(job.sstate.diagnosticindices, :accept)
        job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
      end
    end
  end

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    if job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0
      rate!(job.sstate.tune)

      if isa(job.tuner, AcceptanceRateMCTuner)
        tune!(job.sstate.tune, job.tuner)
      end

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
    if job.sstate.count <= job.tuner.nadapt
      tune!(job.sstate.tune, job.tuner, job.sstate.count, job.sstate.a)

      if job.tuner.verbose
        if mod(job.sstate.tune.proposed, job.tuner.period) == 0
          rate!(job.sstate.tune)

          println(
            "Burnin iteration ",
            job.fmt_iter(job.sstate.tune.totproposed),
            " of ",
            job.tuner.nadapt,
            ": ",
            job.fmt_perc(100*job.sstate.tune.rate),
            " % acceptance rate"
          )

          reset_burnin!(job.sstate.tune)
        end
      end
    else
      job.sstate.tune.step = job.sstate.tune.εbar
    end
  end
end
