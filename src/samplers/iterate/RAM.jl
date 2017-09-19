function iterate!(job::BasicMCJob, ::Type{RAM}, ::Type{Univariate})
  job.sstate.count += 1

  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.randnsample = randn()
  job.sstate.pstate.value = job.pstate.value+job.sstate.S*job.sstate.randnsample

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  if (job.sstate.ratio > 0 || (job.sstate.ratio > log(rand())))
    job.pstate.value = job.sstate.pstate.value

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

    if job.tuner.verbose
      job.sstate.tune.accepted += 1
    end
  else
    if !isempty(job.sstate.diagnosticindices)
      if haskey(job.sstate.diagnosticindices, :accept)
        job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
      end
    end
  end

  if job.tuner.verbose
    if job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0
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

  job.sstate.η = min(1, job.sstate.count^(-job.sampler.γ))
  job.sstate.SST = job.sstate.η*(min(1, exp(job.sstate.ratio))-job.sampler.targetrate)
  job.sstate.SST = abs2(job.sstate.S)*(1+job.sstate.SST)
  job.sstate.S = chol(job.sstate.SST)
end

function iterate!(job::BasicMCJob, ::Type{RAM}, ::Type{Multivariate})
  job.sstate.count += 1

  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.randnsample[:] = randn(job.pstate.size)
  job.sstate.pstate.value[:] = job.pstate.value+job.sstate.S*job.sstate.randnsample

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  if (job.sstate.ratio > 0 || (job.sstate.ratio > log(rand())))
    job.pstate.value = copy(job.sstate.pstate.value)

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

    if job.tuner.verbose
      job.sstate.tune.accepted += 1
    end
  else
    if !isempty(job.sstate.diagnosticindices)
      if haskey(job.sstate.diagnosticindices, :accept)
        job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
      end
    end
  end

  if job.tuner.verbose
    if job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0
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

  job.sstate.η = min(1, job.pstate.size*job.sstate.count^(-job.sampler.γ))
  job.sstate.SST[:, :] = (
    job.sstate.randnsample*job.sstate.randnsample'/dot(job.sstate.randnsample, job.sstate.randnsample)*
    job.sstate.η*(min(1, exp(job.sstate.ratio))-job.sampler.targetrate)
  )
  job.sstate.SST[:, :] = job.sstate.S*(eye(job.pstate.size)+job.sstate.SST)*job.sstate.S'
  job.sstate.S[:, :] = ctranspose(chol(Hermitian(job.sstate.SST)))
end
