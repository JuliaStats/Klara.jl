function codegen_iterate_mala(job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  burninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in MALA code generation")
  end

  stepsize = isa(job.tuner, AcceptanceRateMCTuner) ? :(_job.sstate.tune.step) : :(_job.sampler.driftstep)

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  push!(body, :(_job.sstate.vmean = _job.pstate.value+0.5*$(stepsize)*_job.pstate.gradlogtarget))

  if vform == Univariate
    push!(body, :(_job.sstate.pstate.value = _job.sstate.vmean+sqrt($(stepsize))*randn()))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.pstate.value = _job.sstate.vmean+sqrt($(stepsize))*randn(_job.pstate.size)))
  end

  push!(body, :(_job.parameter.uptogradlogtarget!(_job.sstate.pstate)))

  if vform == Univariate
    push!(
      body,
      :(_job.sstate.pnewgivenold = -0.5*(abs2(_job.sstate.vmean-_job.sstate.pstate.value)/$(stepsize)+log(2*pi*$(stepsize))))
    )
  elseif vform == Multivariate
    push!(
      body,
      :(
        _job.sstate.pnewgivenold =
        sum(-0.5*(abs2(_job.sstate.vmean-_job.sstate.pstate.value)/$(stepsize)+log(2*pi*$(stepsize))))
      )
    )
  end

  push!(body, :(_job.sstate.vmean = _job.sstate.pstate.value+0.5*$(stepsize)*_job.sstate.pstate.gradlogtarget))

  if vform == Univariate
    push!(
      body,
      :(_job.sstate.poldgivennew = -0.5*(abs2(_job.sstate.vmean-_job.pstate.value)/$(stepsize)+log(2*pi*$(stepsize))))
    )
  elseif vform == Multivariate
    push!(
      body,
      :(_job.sstate.poldgivennew = sum(-0.5*(abs2(_job.sstate.vmean-_job.pstate.value)/$(stepsize)+log(2*pi*$(stepsize)))))
    )
  end

  push!(
    body,
    :(
      _job.sstate.ratio =
      _job.sstate.pstate.logtarget+_job.sstate.poldgivennew-_job.pstate.logtarget-_job.sstate.pnewgivenold
    )
  )

  if vform == Univariate
    push!(update, :(_job.pstate.value = _job.sstate.pstate.value))
    push!(update, :(_job.pstate.gradlogtarget = _job.sstate.pstate.gradlogtarget))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :(_job.pstate.gradloglikelihood = _job.sstate.pstate.gradloglikelihood))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :(_job.pstate.gradlogprior = _job.sstate.pstate.gradlogprior))
    end
  elseif vform == Multivariate
    push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))
    push!(update, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstate.gradlogtarget)))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstate.gradloglikelihood)))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :(_job.pstate.gradlogprior = copy(_job.sstate.pstate.gradlogprior)))
    end
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

  push!(body, Expr(:if, :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))), Expr(:block, update...), noupdate...))

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune)))

    if isa(job.tuner, AcceptanceRateMCTuner)
      push!(burninbody, :(tune!(_job.sstate.tune, _job.tuner)))
    end

    if job.tuner.verbose
      fmt_iter = format_iteration(ndigits(job.range.burnin))
      fmt_perc = format_percentage()

      push!(burninbody, :(println(
        "Burnin iteration ",
        $(fmt_iter)(_job.sstate.tune.totproposed),
        " of ",
        _job.range.burnin,
        ": ",
        $(fmt_perc)(100*_job.sstate.tune.rate),
        " % acceptance rate"
      )))
    end

    push!(burninbody, :(reset_burnin!(_job.sstate.tune)))

    push!(
      body,
      Expr(
        :if,
        :(_job.sstate.tune.totproposed <= _job.range.burnin && mod(_job.sstate.tune.proposed, _job.tuner.period) == 0),
        Expr(:block, burninbody...)
      )
    )
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym iterate_mala

  result = quote
    function $iterate_mala(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end
