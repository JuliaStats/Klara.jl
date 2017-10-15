function iterate!(job::BasicMCJob, ::Type{UnvAMWG}, ::Type{Univariate})
  job.sstate.tune.proposed += 1

  if mod(job.sstate.tune.proposed, job.tuner.period) == 0
    set_batch!(job.sstate.tune, job.tuner)
    set_delta!(job.sstate.tune, job.tuner)
  end

  if isinf(job.sampler.lower) && isinf(job.sampler.upper)
    job.sstate.pstate.value = job.pstate.value+sqrt(exp(job.sstate.tune.logσ))*randn()
    job.parameter.logtarget!(job.sstate.pstate)

    job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget
  else
    job.sstate.proposal = Truncated(Normal(
      job.pstate.value, sqrt(exp(job.sstate.tune.logσ))), job.sampler.lower, job.sampler.upper
    )
    job.sstate.pstate.value = rand(job.sstate.proposal)
    job.parameter.logtarget!(job.sstate.pstate)

    job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget
    job.sstate.ratio -= logpdf(job.sstate.proposal, job.sstate.pstate.value)

    job.sstate.proposal = Truncated(Normal(
      job.sstate.pstate.value, sqrt(exp(job.sstate.tune.logσ))), job.sampler.lower, job.sampler.upper
    )

    job.sstate.ratio += logpdf(job.sstate.proposal, job.pstate.value)
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

    job.sstate.tune.accepted += 1
  else
    if !isempty(job.sstate.diagnosticindices)
      if haskey(job.sstate.diagnosticindices, :accept)
        job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
      end
    end
  end

  if haskey(job.sstate.diagnosticindices, :logσ)
    job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:logσ]] = job.sstate.tune.logσ
  end

  if mod(job.sstate.tune.proposed, job.tuner.period) == 0
    rate!(job.sstate.tune)

    if job.tuner.verbose && job.sstate.tune.totproposed <= job.range.burnin
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

    tune!(job.sstate.tune, job.tuner)
    reset!(job.sstate.tune)
  end
end

function iterate!(job::BasicMCJob, ::Type{MuvAMWG}, ::Type{Multivariate})
  job.sstate.tune.proposed += 1

  if mod(job.sstate.tune.proposed, job.tuner.period) == 0
    set_batch!(job.sstate.tune, job.tuner)
    set_delta!(job.sstate.tune, job.tuner)

    if job.tuner.verbose && job.sstate.tune.totproposed <= job.range.burnin
      println("Burnin iteration ", job.fmt_iter(job.sstate.tune.totproposed), " of ", job.range.burnin, ":")
    end
  end

  for i in 1:job.pstate.size
    if isinf(job.sampler.lower[i]) && isinf(job.sampler.upper[i])
      job.sstate.pstate.value[i] = job.pstate.value[i]+sqrt(exp(job.sstate.tune.logσ[i]))*randn()
      job.parameter.logtarget!(job.sstate.pstate)

      job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget
    else
      job.sstate.proposal = Truncated(Normal(
        job.pstate.value[i], sqrt(exp(job.sstate.tune.logσ[i]))), job.sampler.lower[i], job.sampler.upper[i]
      )
      job.sstate.pstate.value[i] = rand(job.sstate.proposal)
      job.parameter.logtarget!(job.sstate.pstate)

      job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget
      job.sstate.ratio -= logpdf(job.sstate.proposal, job.sstate.pstate.value)

      job.sstate.proposal = Truncated(Normal(
        job.sstate.pstate.value[i], sqrt(exp(job.sstate.tune.logσ[i]))), job.sampler.lower[i], job.sampler.upper[i]
      )

      job.sstate.ratio += logpdf(job.sstate.proposal, job.pstate.value)
    end

    if (job.sstate.ratio > 0 || (job.sstate.ratio > log(rand())))
      job.pstate.value[i] = job.sstate.pstate.value[i]

      job.pstate.logtarget = job.sstate.pstate.logtarget

      if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
        job.pstate.loglikelihood = job.sstate.pstate.loglikelihood
      end
      if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
        job.pstate.logprior = job.sstate.pstate.logprior
      end

      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]][i] = true
        end
      end

      job.sstate.tune.accepted[i] += 1
    else
      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]][i] = false
        end
      end
    end

    if haskey(job.sstate.diagnosticindices, :logσ)
      job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:logσ]][i] = job.sstate.tune.logσ[i]
    end

    if mod(job.sstate.tune.proposed, job.tuner.period) == 0
      rate!(job.sstate.tune, i)

      if job.tuner.verbose && job.sstate.tune.totproposed <= job.range.burnin
        println(
          "  parameter ",
          i,
          " of ",
          job.pstate.size,
          ": ",
          job.fmt_perc(100*job.sstate.tune.rate[i]),
          " % acceptance rate"
        )
      end

      tune!(job.sstate.tune, job.tuner, i)
      reset!(job.sstate.tune, i)
    end
  end

  if mod(job.sstate.tune.proposed, job.tuner.period) == 0
    reset!(job.sstate.tune)
  end
end
