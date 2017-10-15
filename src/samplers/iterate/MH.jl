function iterate!(job::BasicMCJob, ::Type{MH}, ::Type{Univariate})
  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.proposal = job.sampler.setproposal(job.pstate)

  job.sstate.pstate.value = rand(job.sstate.proposal)

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  if !(job.sampler.symmetric)
    job.sstate.ratio -= logpdf(job.sstate.proposal, job.sstate.pstate.value)
    if !(job.sampler.normalised)
      job.sstate.ratio -= lognormalise(job.sstate.proposal)
    end
    job.sstate.proposal = job.sampler.setproposal(job.sstate.pstate)
    job.sstate.ratio += logpdf(job.sstate.proposal, job.pstate.value)
    if !(job.sampler.normalised)
      job.sstate.ratio += lognormalise(job.sstate.proposal)
    end
  end

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
end

function iterate!(job::BasicMCJob, ::Type{MH}, ::Type{Multivariate})
  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.proposal = job.sampler.setproposal(job.pstate)

  job.sstate.pstate.value[:] =  rand(job.sstate.proposal)

  job.parameter.logtarget!(job.sstate.pstate)

  job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

  if !(job.sampler.symmetric)
    job.sstate.ratio -= logpdf(job.sstate.proposal, job.sstate.pstate.value)
    if !(job.sampler.normalised)
      job.sstate.ratio -= lognormalise(job.sstate.proposal)
    end
    job.sstate.proposal = job.sampler.setproposal(job.sstate.pstate)
    job.sstate.ratio += logpdf(job.sstate.proposal, job.pstate.value)
    if !(job.sampler.normalised)
      job.sstate.ratio += lognormalise(job.sstate.proposal)
    end
  end

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
end
