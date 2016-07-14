function codegen(::Type{Val{:iterate}}, ::Type{MALA}, job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  burninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in MALA code generation")
  end

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  push!(body, :(_job.sstate.μ = _job.pstate.value+0.5*_job.sstate.tune.step*_job.pstate.gradlogtarget))

  if vform == Univariate
    push!(body, :(_job.sstate.pstate.value = _job.sstate.μ+sqrt(_job.sstate.tune.step)*randn()))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.pstate.value = _job.sstate.μ+sqrt(_job.sstate.tune.step)*randn(_job.pstate.size)))
  end

  push!(body, :(_job.parameter.uptogradlogtarget!(_job.sstate.pstate)))

  push!(body, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  if vform == Univariate
    push!(body, :(_job.sstate.ratio += 0.5*(abs2(_job.sstate.μ-_job.sstate.pstate.value)/_job.sstate.tune.step)))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.ratio += sum(0.5*(abs2(_job.sstate.μ-_job.sstate.pstate.value)/_job.sstate.tune.step))))
  end

  push!(body, :(_job.sstate.μ = _job.sstate.pstate.value+0.5*_job.sstate.tune.step*_job.sstate.pstate.gradlogtarget))

  if vform == Univariate
    push!(body, :(_job.sstate.ratio -= 0.5*(abs2(_job.sstate.μ-_job.pstate.value)/_job.sstate.tune.step)))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.ratio -= sum(0.5*(abs2(_job.sstate.μ-_job.pstate.value)/_job.sstate.tune.step))))
  end

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
  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
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

  @gensym _iterate

  result = quote
    function $_iterate(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end
