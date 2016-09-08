function codegen(::Type{Val{:iterate}}, ::Type{HMC}, job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  burninbody = []
  ifburninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in HMC code generation")
  end

  if isa(job.tuner, DualAveragingMCTuner)
    push!(body, :(_job.sstate.count += 1))
  end

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) ||
    isa(job.tuner, AcceptanceRateMCTuner) ||
    (isa(job.tuner, DualAveragingMCTuner) && job.tuner.verbose)
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  if vform == Univariate
    push!(body, :(_job.sstate.momentum = randn()))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.momentum = randn(_job.pstate.size)))
  end

  push!(body, :(_job.sstate.oldhamiltonian = hamiltonian(_job.pstate.logtarget, _job.sstate.momentum)))

  if vform == Univariate
    push!(body, :(_job.sstate.pstate.value = _job.pstate.value))
    push!(body, :(_job.sstate.pstate.gradlogtarget = _job.pstate.gradlogtarget))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(body, :(_job.sstate.pstate.gradloglikelihood = _job.pstate.gradloglikelihood))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(body, :(_job.sstate.pstate.gradlogprior = _job.pstate.gradlogprior))
    end
  elseif vform == Multivariate
    push!(body, :(_job.sstate.pstate.value = copy(_job.pstate.value)))
    push!(body, :(_job.sstate.pstate.gradlogtarget = copy(_job.pstate.gradlogtarget)))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(body, :(_job.sstate.pstate.gradloglikelihood = copy(_job.pstate.gradloglikelihood)))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(body, :(_job.sstate.pstate.gradlogprior = copy(_job.pstate.gradlogprior)))
    end
  end

  if isa(job.tuner, DualAveragingMCTuner)
    push!(body, :(_job.sstate.nleaps = max(1, Int(round(_job.sstate.tune.λ/_job.sstate.tune.step)))))
  end

  push!(body, :(
    for i in 1:_job.sstate.nleaps
      leapfrog!(_job.sstate, _job.parameter)
    end
  ))

  push!(body, :(_job.parameter.logtarget!(_job.sstate.pstate)))

  push!(body, :(_job.sstate.newhamiltonian = hamiltonian(_job.sstate.pstate.logtarget, _job.sstate.momentum)))

  push!(body, :(_job.sstate.ratio = _job.sstate.newhamiltonian-_job.sstate.oldhamiltonian))

  push!(body, :(_job.sstate.a = min(1., exp(_job.sstate.ratio))))

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
  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) ||
    isa(job.tuner, AcceptanceRateMCTuner) ||
    (isa(job.tuner, DualAveragingMCTuner) && job.tuner.verbose)
    push!(update, :(_job.sstate.tune.accepted += 1))
  end

  push!(body, Expr(:if, :(rand() < _job.sstate.a), Expr(:block, update...), noupdate...))

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
  elseif isa(job.tuner, DualAveragingMCTuner)
    push!(burninbody, :(tune!(_job.sstate.tune, _job.tuner, _job.sstate.count, _job.sstate.a)))

    if job.tuner.verbose
      fmt_iter = format_iteration(ndigits(job.tuner.nadapt))
      fmt_perc = format_percentage()

      push!(ifburninbody, :(rate!(_job.sstate.tune)))

      push!(ifburninbody, :(println(
        "Burnin iteration ",
        $(fmt_iter)(_job.sstate.tune.totproposed),
        " of ",
        _job.tuner.nadapt,
        ": ",
        $(fmt_perc)(100*_job.sstate.tune.rate),
        " % acceptance rate"
      )))

      push!(ifburninbody, :(reset_burnin!(_job.sstate.tune)))

      push!(burninbody, Expr(:if, :(mod(_job.sstate.tune.proposed, _job.tuner.period) == 0), Expr(:block, ifburninbody...)))
    end

    push!(
      body,
      Expr(
        :if,
        :(_job.sstate.count <= _job.tuner.nadapt),
        Expr(:block, burninbody...),
        :(_job.sstate.tune.step = _job.sstate.tune.εbar)
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
