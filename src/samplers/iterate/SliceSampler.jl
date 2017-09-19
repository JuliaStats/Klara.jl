function iterate!(job::BasicMCJob, ::Type{SliceSampler}, ::Type{Univariate})
  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  job.sstate.loguprime = log(rand())+job.pstate.logtarget
  job.sstate.lstate.value = job.pstate.value
  job.sstate.rstate.value = job.pstate.value
  job.sstate.primestate.value = job.pstate.value

  job.sstate.runiform = rand()
  job.sstate.lstate.value = job.pstate.value-job.sstate.runiform*job.sampler.widths[1]
  job.sstate.rstate.value = job.pstate.value+(1-job.sstate.runiform)*job.sampler.widths[1]

  if job.sampler.stepout
    job.parameter.logtarget!(job.sstate.lstate)

    while job.sstate.lstate.logtarget > job.sstate.loguprime
      job.sstate.lstate.value -= job.sampler.widths[1]
      job.parameter.logtarget!(job.sstate.lstate)
    end

    job.parameter.logtarget!(job.sstate.rstate)

    while job.sstate.rstate.logtarget > job.sstate.loguprime
      job.sstate.rstate.value += job.sampler.widths[1]
      job.parameter.logtarget!(job.sstate.rstate)
    end
  end

  while true
    job.sstate.primestate.value =
      rand()*(job.sstate.rstate.value-job.sstate.lstate.value)+job.sstate.lstate.value
    job.pstate.logtarget = job.parameter.logtarget!(job.sstate.primestate)
    if job.pstate.logtarget > job.sstate.loguprime
      break
    else
      if job.sstate.primestate.value > job.pstate.value
        job.sstate.rstate.value = job.sstate.primestate.value
      elseif job.sstate.primestate.value < job.pstate.value
        job.sstate.lstate.value = job.sstate.primestate.value
      else
        @assert false "Shrunk to current position and still not acceptable"
      end
    end
  end

  job.pstate.value = job.sstate.primestate.value

  if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
    if job.tuner.verbose
      println("Burnin iteration ", job.fmt_iter(job.sstate.tune.totproposed), " of ", job.range.burnin)
    end

    job.sstate.tune.totproposed += job.sstate.tune.proposed
    job.sstate.tune.proposed = 0
  end
end

function iterate!(job::BasicMCJob, ::Type{SliceSampler}, ::Type{Multivariate})
  if job.tuner.verbose
    job.sstate.tune.proposed += 1
  end

  for i = 1:job.pstate.size
    job.sstate.loguprime = log(rand())+job.pstate.logtarget
    job.sstate.lstate.value = copy(job.pstate.value)
    job.sstate.rstate.value = copy(job.pstate.value)
    job.sstate.primestate.value = copy(job.pstate.value)

    job.sstate.runiform = rand()
    job.sstate.lstate.value[i] = job.pstate.value[i]-job.sstate.runiform*job.sampler.widths[i]
    job.sstate.rstate.value[i] = job.pstate.value[i]+(1-job.sstate.runiform)*job.sampler.widths[i]

    if job.sampler.stepout
      job.parameter.logtarget!(job.sstate.lstate)

      while job.sstate.lstate.logtarget > job.sstate.loguprime
        job.sstate.lstate.value[i] -= job.sampler.widths[i]
        job.parameter.logtarget!(job.sstate.lstate)
      end

      job.parameter.logtarget!(job.sstate.rstate)

      while job.sstate.rstate.logtarget > job.sstate.loguprime
        job.sstate.rstate.value[i] += job.sampler.widths[i]
        job.parameter.logtarget!(job.sstate.rstate)
      end
    end

    while true
      job.sstate.primestate.value[i] =
        rand()*(job.sstate.rstate.value[i]-job.sstate.lstate.value[i])+job.sstate.lstate.value[i]
      job.pstate.logtarget = job.parameter.logtarget!(job.sstate.primestate)
      if job.pstate.logtarget > job.sstate.loguprime
        break
      else
        if job.sstate.primestate.value[i] > job.pstate.value[i]
          job.sstate.rstate.value[i] = job.sstate.primestate.value[i]
        elseif job.sstate.primestate.value[i] < job.pstate.value[i]
          job.sstate.lstate.value[i] = job.sstate.primestate.value[i]
        else
          @assert false "Shrunk to current position and still not acceptable"
        end
      end
    end

    job.pstate.value[i] = job.sstate.primestate.value[i]
  end

  if (job.sstate.tune.totproposed <= job.range.burnin && mod(job.sstate.tune.proposed, job.tuner.period) == 0)
    if job.tuner.verbose
      println("Burnin iteration ", job.fmt_iter(job.sstate.tune.totproposed), " of ", job.range.burnin)
    end

    job.sstate.tune.totproposed += job.sstate.tune.proposed
    job.sstate.tune.proposed = 0
  end
end
