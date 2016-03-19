function codegen(::Type{Val{:iterate}}, ::Type{ARS}, job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in ARS code generation")
  end

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  if vform == Univariate
    push!(body, :(_job.sstate.pstate.value = _job.pstate.value+_job.sampler.jumpscale*randn()))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.pstate.value = _job.pstate.value+_job.sampler.jumpscale*randn(_job.pstate.size)))
  end

  push!(body, :(_job.parameter.logtarget!(_job.sstate.pstate)))

  push!(body, :(_job.sstate.logproposal = _job.sampler.logproposal(_job.sstate.pstate.value)))

  push!(body, :(_job.sstate.weight = _job.sstate.pstate.logtarget-_job.sampler.proposalscale-_job.sstate.logproposal))

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
  if in(:accept, job.outopts[:diagnostics])
    push!(update, :(_job.pstate.diagnosticvalues[1] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[1] = false))
  end
  if job.tuner.verbose
    push!(update, :(_job.sstate.tune.accepted += 1))
  end

  push!(body, Expr(:if, :(_job.sstate.weight > log(rand())), Expr(:block, update...), noupdate...))

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
