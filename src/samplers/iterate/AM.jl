function codegen(::Type{Val{:iterate}}, ::Type{AM}, job::BasicMCJob)
  local result::Expr
  update = []
  noupdate = []
  body = []
  local dindex::Integer

  vform = variate_form(job.pstate)
  if vform != Multivariate
    error("Only multivariate parameter states allowed in MH code generation")
  end

  push!(body, :(_job.sstate.count += 1))

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  push!(
    body,
    :(
      covariance!(
        _job.sstate.C,
        _job.sstate.C,
        _job.sstate.count-2,
        _job.pstate.value,
        _job.sstate.lastmean,
        _job.sstate.secondlastmean
        )
    )
  )

  push!(body, :(setproposal!(_job.sstate, _job.sampler, _job.pstate)))

  push!(body, :(_job.sstate.pstate.value[:] =  rand(_job.sstate.proposal)))

  push!(body, :(_job.parameter.logtarget!(_job.sstate.pstate)))

  push!(body, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(body, :(_job.sstate.ratio -= logpdf(_job.sstate.proposal, _job.sstate.pstate.value)))

  push!(body, :(setproposal!(_job.sstate, _job.sampler, _job.sstate.pstate)))

  push!(body, :(_job.sstate.ratio += logpdf(_job.sstate.proposal, _job.pstate.value)))

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))
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

  push!(body, :(_job.sstate.secondlastmean = copy(_job.sstate.lastmean)))

  push!(body, :(mean!(_job.sstate.lastmean, _job.sstate.count, _job.pstate.value)))

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
