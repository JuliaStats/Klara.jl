function iterate!(job::BasicMCJob, ::Type{SMMALA}, ::Type{Univariate})
  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    job.sstate.tune.proposed += 1
  end

  job.sstate.μ = job.pstate.value+0.5*job.sstate.tune.step*job.sstate.oldfirstterm
  job.sstate.pstate.value = job.sstate.μ+job.sstate.sqrttunestep*job.sstate.cholinvtensor*randn()

  job.parameter.uptotensorlogtarget!(job.sstate.pstate)

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  job.sstate.ratio += (
    0.5*(
      log(job.sstate.tune.step*job.sstate.oldinvtensor)
      +abs2(job.sstate.pstate.value-job.sstate.μ)*job.pstate.tensorlogtarget/job.sstate.tune.step
    )
  )

  job.sstate.newinvtensor = inv(job.sstate.pstate.tensorlogtarget)

  job.sstate.newfirstterm = job.sstate.newinvtensor*job.sstate.pstate.gradlogtarget

  job.sstate.μ = job.sstate.pstate.value+0.5*job.sstate.tune.step*job.sstate.newfirstterm

  job.sstate.ratio -= (
    0.5*(
      log(job.sstate.tune.step*job.sstate.newinvtensor)
      +abs2(job.pstate.value-job.sstate.μ)*job.sstate.pstate.tensorlogtarget/job.sstate.tune.step
    )
  )

  if (job.sstate.ratio > 0 || (job.sstate.ratio > log(rand())))
    job.pstate.value = job.sstate.pstate.value

    job.pstate.gradlogtarget = job.sstate.pstate.gradlogtarget
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      job.pstate.gradloglikelihood = job.sstate.pstate.gradloglikelihood
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      job.pstate.gradlogprior = job.sstate.pstate.gradlogprior
    end

    job.pstate.tensorlogtarget = job.sstate.pstate.tensorlogtarget
    if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
      job.pstate.tensorloglikelihood = job.sstate.pstate.tensorloglikelihood
    end
    if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
      job.pstate.tensorlogprior = job.sstate.pstate.tensorlogprior
    end

    job.sstate.oldinvtensor = job.sstate.newinvtensor
    job.sstate.cholinvtensor = chol(job.sstate.newinvtensor)
    job.sstate.oldfirstterm = job.sstate.newfirstterm

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
        job.sstate.sqrttunestep = sqrt(job.sstate.tune.step)
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

function iterate!(job::BasicMCJob, ::Type{SMMALA}, ::Type{Multivariate})
  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    job.sstate.tune.proposed += 1
  end

  job.sstate.μ[:] = job.pstate.value+0.5*job.sstate.tune.step*job.sstate.oldfirstterm
  job.sstate.pstate.value[:] = job.sstate.μ+job.sstate.sqrttunestep*job.sstate.cholinvtensor*randn(job.pstate.size)

  job.parameter.uptotensorlogtarget!(job.sstate.pstate)

  if job.sampler.transform != nothing
    job.sstate.pstate.tensorlogtarget[:, :] = job.sampler.transform(job.sstate.pstate.tensorlogtarget)
  end

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  job.sstate.ratio += (
    0.5*(
      logdet(job.sstate.tune.step*job.sstate.oldinvtensor)
      +dot(
        job.sstate.pstate.value-job.sstate.μ,
        job.pstate.tensorlogtarget*(job.sstate.pstate.value-job.sstate.μ)
      )/job.sstate.tune.step
    )
  )

  job.sstate.newinvtensor[:, :] = inv(job.sstate.pstate.tensorlogtarget)

  job.sstate.newfirstterm[:] = job.sstate.newinvtensor*job.sstate.pstate.gradlogtarget

  job.sstate.μ[:] = job.sstate.pstate.value+0.5*job.sstate.tune.step*job.sstate.newfirstterm

  job.sstate.ratio -= (
    0.5*(
      logdet(job.sstate.tune.step*job.sstate.newinvtensor)
      +dot(
        job.pstate.value-job.sstate.μ,
        job.sstate.pstate.tensorlogtarget*(job.pstate.value-job.sstate.μ)
      )/job.sstate.tune.step
    )
  )

  if (job.sstate.ratio > 0 || (job.sstate.ratio > log(rand())))
    job.pstate.value = copy(job.sstate.pstate.value)

    job.pstate.gradlogtarget = copy(job.sstate.pstate.gradlogtarget)
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      job.pstate.gradloglikelihood = copy(job.sstate.pstate.gradloglikelihood)
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      job.pstate.gradlogprior = copy(job.sstate.pstate.gradlogprior)
    end

    job.pstate.tensorlogtarget = copy(job.sstate.pstate.tensorlogtarget)
    if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
      job.pstate.tensorloglikelihood = copy(job.sstate.pstate.tensorloglikelihood)
    end
    if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
      job.pstate.tensorlogprior = copy(job.sstate.pstate.tensorlogprior)
    end

    job.sstate.oldinvtensor = copy(job.sstate.newinvtensor)
    job.sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(job.sstate.newinvtensor)))
    job.sstate.oldfirstterm = copy(job.sstate.newfirstterm)

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
        job.sstate.sqrttunestep = sqrt(job.sstate.tune.step)
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
