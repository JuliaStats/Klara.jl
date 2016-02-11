function codegen_iterate_hmc(job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  burninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in HMC code generation")
  end

  stepsize = isa(job.tuner, AcceptanceRateMCTuner) ? :($(job).sstate.tune.step) : :($(job).sstate.leapstep)

  if job.tuner.verbose
    push!(body, :($(job).sstate.tune.proposed += 1))
  end

  if vform == Univariate
    push!(body, :($(job).sstate.momentum = randn()))
  elseif vform == Multivariate
    push!(body, :($(job).sstate.momentum = randn($(job).pstate.size)))
  end

  push!(body, :($(job).sstate.oldhamiltonian = hamiltonian($(job).pstate.logtarget, $(job).sstate.momentum)))

  if vform == Univariate
    push!(body, :($(job).sstate.pstate.value = $(job).pstate.value))
    push!(body, :($(job).sstate.pstate.gradlogtarget = $(job).pstate.gradlogtarget))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(body, :($(job).sstate.pstate.gradloglikelihood = $(job).pstate.gradloglikelihood))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(body, :($(job).sstate.pstate.gradlogprior = $(job).pstate.gradlogprior))
    end
  elseif vform == Multivariate
    push!(body, :($(job).sstate.pstate.value = copy($(job).pstate.value)))
    push!(body, :($(job).sstate.pstate.gradlogtarget = copy($(job).pstate.gradlogtarget)))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(body, :($(job).sstate.pstate.gradloglikelihood = copy($(job).pstate.gradloglikelihood)))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(body, :($(job).sstate.pstate.gradlogprior = copy($(job).pstate.gradlogprior)))
    end
  end

  push!(body, :(
    for i in 1:$(job).sampler.nleaps
      leapfrog!($(job).sstate, $(job).parameter)
    end
  ))

  push!(body, :($(job).parameter.logtarget!($(job).sstate.pstate)))

  push!(body, :($(job).sstate.newhamiltonian = hamiltonian($(job).sstate.pstate.logtarget, $(job).sstate.momentum)))

  push!(body, :($(job).sstate.ratio = $(job).sstate.oldhamiltonian-$(job).sstate.newhamiltonian))

  if vform == Univariate
    push!(update, :($(job).pstate.value = $(job).sstate.pstate.value))
    push!(update, :($(job).pstate.gradlogtarget = $(job).sstate.pstate.gradlogtarget))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :($(job).pstate.gradloglikelihood = $(job).sstate.pstate.gradloglikelihood))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :($(job).pstate.gradlogprior = $(job).sstate.pstate.gradlogprior))
    end
  elseif vform == Multivariate
    push!(update, :($(job).pstate.value = copy($(job).sstate.pstate.value)))
    push!(update, :($(job).pstate.gradlogtarget = copy($(job).sstate.pstate.gradlogtarget)))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :($(job).pstate.gradloglikelihood = copy($(job).sstate.pstate.gradloglikelihood)))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :($(job).pstate.gradlogprior = copy($(job).sstate.pstate.gradlogprior)))
    end
  end
  push!(update, :($(job).pstate.logtarget = $(job).sstate.pstate.logtarget))
  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :($(job).pstate.loglikelihood = $(job).sstate.pstate.loglikelihood))
  end
  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :($(job).pstate.logprior = $(job).sstate.pstate.logprior))
  end
  if in(:accept, job.outopts[:diagnostics])
    push!(update, :($(job).pstate.diagnosticvalues[1] = true))
    push!(noupdate, :($(job).pstate.diagnosticvalues[1] = false))
  end
  if job.tuner.verbose
    push!(update, :($(job).sstate.tune.accepted += 1))
  end

  push!(
    body,
    Expr(:if, :($(job).sstate.ratio > 0 || ($(job).sstate.ratio > log(rand()))), Expr(:block, update...), noupdate...)
  )

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!($(job).sstate.tune)))

    if isa(job.tuner, AcceptanceRateMCTuner)
      push!(burninbody, :(tune!($(job).sstate.tune, $(job).tuner)))
    end

    if job.tuner.verbose
      fmt_iter = format_iteration(ndigits(job.range.burnin))
      fmt_perc = format_percentage()

      push!(burninbody, :(println(
        "Burnin iteration ",
        $(fmt_iter)($(job).sstate.tune.totproposed),
        " of ",
        $(job).range.burnin,
        ": ",
        $(fmt_perc)(100*$(job).sstate.tune.rate),
        " % acceptance rate"
      )))
    end

    push!(burninbody, :(reset_burnin!($(job).sstate.tune)))

    push!(body, Expr(
      :if,
      :($(job).sstate.tune.totproposed <= $(job).range.burnin && mod($(job).sstate.tune.proposed, $(job).tuner.period) == 0),
      Expr(:block, burninbody...)
    ))
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym iterate_hmc

  result = quote
    function $iterate_hmc()
      $(body...)
    end
  end

  result
end
