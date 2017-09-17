function iterate!(job::BasicMCJob, ::Type{ARS}, ::Type{Univariate})
  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.pstate.value = job.pstate.value+job.sampler.jumpscale*randn()

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.logproposal = job.sampler.logproposal(job.sstate.pstate.value)

  job.sstate.weight = job.sstate.pstate.logtarget-job.sampler.proposalscale-job.sstate.logproposal

  if (job.sstate.weight > log(rand()))
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
        job.fmt_perc(100*_job.sstate.tune.rate),
        " % acceptance rate"
      )
      reset_burnin!(job.sstate.tune)
    end
  end
end

function iterate!(job::BasicMCJob, ::Type{ARS}, ::Type{Multivariate})
  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.pstate.value[:] = job.pstate.value+job.sampler.jumpscale*randn(job.pstate.size)

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.logproposal = job.sampler.logproposal(job.sstate.pstate.value)

  job.sstate.weight = job.sstate.pstate.logtarget-job.sampler.proposalscale-job.sstate.logproposal

  if (job.sstate.weight > log(rand()))
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
        job.fmt_perc(100*_job.sstate.tune.rate),
        " % acceptance rate"
      )
      reset_burnin!(job.sstate.tune)
    end
  end
end
