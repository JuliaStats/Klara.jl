function codegen(::Type{Val{:iterate}}, ::Type{RAM}, job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  body = []
  dindex::Integer

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in MH code generation")
  end

  push!(body, :(_job.sstate.count += 1))

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  if vform == Univariate
    push!(body, :(_job.sstate.randnsample = randn()))
    push!(body, :(_job.sstate.pstate.value = _job.pstate.value+_job.sstate.S*_job.sstate.randnsample))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.randnsample[:] = randn(_job.pstate.size)))
    push!(body, :(_job.sstate.pstate.value[:] = _job.pstate.value+_job.sstate.S*_job.sstate.randnsample))
  end

  push!(body, :(_job.parameter.logtarget!(_job.sstate.pstate)))

  push!(body, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  if vform == Univariate
    push!(update, :(_job.pstate.value = _job.sstate.pstate.value))
  elseif vform == Multivariate
    push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))
  end
  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))
  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end
  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end
  dindex = findfirst(job.outopts[:diagnostics], :accept)
  if dindex != 0
    push!(update, :(_job.pstate.diagnosticvalues[$dindex] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[$dindex] = false))
  end
  if job.tuner.verbose
    push!(update, :(_job.sstate.tune.accepted += 1))
  end

  push!(body, Expr(:if, :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))), Expr(:block, update...), noupdate...))

  if job.tuner.verbose
    fmt_iter = format_iteration(ndigits(job.range.burnin))
    fmt_perc = format_percentage()

    push!(body, :(
      if _job.sstate.tune.totproposed <= _job.range.burnin && mod(_job.sstate.tune.proposed, _job.tuner.period) == 0
        rate!(_job.sstate.tune)
        println(
          "Burnin iteration ",
          $(fmt_iter)(_job.sstate.tune.totproposed),
          " of ",
          _job.range.burnin,
          ": ",
          $(fmt_perc)(100*_job.sstate.tune.rate),
          " % acceptance rate"
        )
        reset_burnin!(_job.sstate.tune)
      end
    ))
  end

  if vform == Univariate
    push!(body, :(_job.sstate.η = min(1, _job.sstate.count^(-_job.sampler.γ))))

    push!(body, :(_job.sstate.SST = _job.sstate.η*(min(1, exp(_job.sstate.ratio))-_job.sampler.targetrate)))

    push!(body, :(_job.sstate.SST = abs2(_job.sstate.S)*(1+_job.sstate.SST)))

    push!(body, :(_job.sstate.S = chol(_job.sstate.SST)))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.η = min(1, _job.pstate.size*_job.sstate.count^(-_job.sampler.γ))))

    push!(
      body,
      :(
        _job.sstate.SST[:, :] = (
          _job.sstate.randnsample*_job.sstate.randnsample'/dot(_job.sstate.randnsample, _job.sstate.randnsample)*
          _job.sstate.η*(min(1, exp(_job.sstate.ratio))-_job.sampler.targetrate)
        )
      )
    )

    push!(body, :(_job.sstate.SST[:, :] = _job.sstate.S*(eye(_job.pstate.size)+_job.sstate.SST)*_job.sstate.S'))

    push!(body, :(_job.sstate.S[:, :] = ctranspose(chol(_job.sstate.SST))))
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym _iterate

  result = quote
    function $_iterate(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end
