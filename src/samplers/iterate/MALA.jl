function iterate!(job::BasicMCJob, ::Type{MALA}, ::Type{Univariate})
  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    job.sstate.tune.proposed += 1
  end

  job.sstate.μ = job.pstate.value+0.5*job.sstate.tune.step*job.pstate.gradlogtarget
  job.sstate.pstate.value = job.sstate.μ+sqrt(job.sstate.tune.step)*randn()

  job.parameter.uptogradlogtarget!(job.sstate.pstate)

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  job.sstate.ratio += 0.5*(abs2(job.sstate.μ-job.sstate.pstate.value)/job.sstate.tune.step)
  job.sstate.μ = job.sstate.pstate.value+0.5*job.sstate.tune.step*job.sstate.pstate.gradlogtarget
  job.sstate.ratio -= 0.5*(abs2(job.sstate.μ-job.pstate.value)/job.sstate.tune.step)

  if (job.sstate.ratio > 0 || (job.sstate.ratio > log(rand())))
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

    if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
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
    if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
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
  end
end

function iterate!(job::BasicMCJob, ::Type{MALA}, ::Type{Multivariate})
  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    job.sstate.tune.proposed += 1
  end

  job.sstate.μ[:] = job.pstate.value+0.5*job.sstate.tune.step*job.pstate.gradlogtarget
  job.sstate.pstate.value[:] = job.sstate.μ+sqrt(job.sstate.tune.step)*randn(job.pstate.size)

  job.parameter.uptogradlogtarget!(job.sstate.pstate)

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  job.sstate.ratio += sum(0.5*(abs2.(job.sstate.μ-job.sstate.pstate.value)/job.sstate.tune.step))
  job.sstate.μ[:] = job.sstate.pstate.value+0.5*job.sstate.tune.step*job.sstate.pstate.gradlogtarget
  job.sstate.ratio -= sum(0.5*(abs2.(job.sstate.μ-job.pstate.value)/job.sstate.tune.step))

  if (job.sstate.ratio > 0 || (job.sstate.ratio > log(rand())))
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

    if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
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
    if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
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
  end
end
