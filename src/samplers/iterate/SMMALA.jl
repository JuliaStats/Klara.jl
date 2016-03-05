function codegen_iterate_smmala(job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  burninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in SMMALA code generation")
  end

  stepsize = isa(job.tuner, AcceptanceRateMCTuner) ? :($(job).sstate.tune.step) : :($(job).sampler.driftstep)

  if job.tuner.verbose
    push!(body, :($(job).sstate.tune.proposed += 1))
  end

  push!(body, :($(job).sstate.vmean = $(job).pstate.value+0.5*$(stepsize)*$(job).sstate.oldfirstterm))

  push!(body, :($(job).sstate.cholinvtensor = chol($(stepsize)*$(job).sstate.oldinvtensor)))

  if vform == Univariate
    push!(body, :($(job).sstate.pstate.value = $(job).sstate.vmean+$(job).sstate.cholinvtensor*randn()))
  elseif vform == Multivariate
    push!(body, :($(job).sstate.pstate.value = $(job).sstate.vmean+$(job).sstate.cholinvtensor'*randn($(job).pstate.size)))
  end

  push!(body, :($(job).parameter.uptotensorlogtarget!($(job).sstate.pstate)))

  if vform == Univariate
    push!(
      body,
      :(
        $(job).sstate.pnewgivenold = (
          -log($(job).sstate.cholinvtensor)
          -0.5*abs2($(job).sstate.vmean-$(job).sstate.pstate.value)*$(job).pstate.tensorlogtarget/$(stepsize)
        )
      )
    )
  elseif vform == Multivariate
    push!(
      body,
      :(
        $(job).sstate.pnewgivenold = (
          -sum(log(diag($(job).sstate.cholinvtensor)))
          -0.5*dot(
            $(job).sstate.vmean-$(job).sstate.pstate.value,
            $(job).pstate.tensorlogtarget*($(job).sstate.vmean-$(job).sstate.pstate.value)
          )/$(stepsize)
        )
      )
    )
  end

  push!(body, :($(job).sstate.newinvtensor = inv($(job).sstate.pstate.tensorlogtarget)))

  push!(body, :($(job).sstate.newfirstterm = $(job).sstate.newinvtensor*$(job).sstate.pstate.gradlogtarget))

  push!(body, :($(job).sstate.vmean = $(job).sstate.pstate.value+0.5*$(stepsize)*$(job).sstate.newfirstterm))

  if vform == Univariate
    push!(
      body,
      :(
        $(job).sstate.poldgivennew = (
          -log(chol($(stepsize)*$(job).sstate.newinvtensor))
          -0.5*abs2($(job).sstate.vmean-$(job).pstate.value)*$(job).sstate.pstate.tensorlogtarget/$(stepsize)
        )
      )
    )
  elseif vform == Multivariate
    push!(
      body,
      :(
        $(job).sstate.poldgivennew = (
          -sum(log(diag(chol($(stepsize)*$(job).sstate.newinvtensor))))
          -0.5*dot(
            $(job).sstate.vmean-$(job).pstate.value,
            $(job).sstate.pstate.tensorlogtarget*($(job).sstate.vmean-$(job).pstate.value)
          )/$(stepsize)
        )
      )
    )
  end

  push!(
    body,
    :(
      $(job).sstate.ratio =
      $(job).sstate.pstate.logtarget+$(job).sstate.poldgivennew-$(job).pstate.logtarget-$(job).sstate.pnewgivenold
    )
  )

  if vform == Univariate
    push!(update, :($(job).pstate.value = $(job).sstate.pstate.value))
    push!(update, :($(job).pstate.gradlogtarget = $(job).sstate.pstate.gradlogtarget))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :($(job).pstate.gradloglikelihood = $(job).sstate.pstate.gradloglikelihood))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :($(job).pstate.gradlogprior = $(job).sstate.pstate.gradlogprior))
    end
    push!(update, :($(job).pstate.tensorlogtarget = $(job).sstate.pstate.tensorlogtarget))
    if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
      push!(update, :($(job).pstate.tensorloglikelihood = $(job).sstate.pstate.tensorloglikelihood))
    end
    if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
      push!(update, :($(job).pstate.tensorlogprior = $(job).sstate.pstate.tensorlogprior))
    end
    push!(update, :($(job).sstate.oldinvtensor = $(job).sstate.newinvtensor))
    push!(update, :($(job).sstate.oldfirstterm = $(job).sstate.newfirstterm))
  elseif vform == Multivariate
    push!(update, :($(job).pstate.value = copy($(job).sstate.pstate.value)))
    push!(update, :($(job).pstate.gradlogtarget = copy($(job).sstate.pstate.gradlogtarget)))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :($(job).pstate.gradloglikelihood = copy($(job).sstate.pstate.gradloglikelihood)))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :($(job).pstate.gradlogprior = copy($(job).sstate.pstate.gradlogprior)))
    end
    push!(update, :($(job).pstate.tensorlogtarget = copy($(job).sstate.pstate.tensorlogtarget)))
    if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
      push!(update, :($(job).pstate.tensorloglikelihood = copy($(job).sstate.pstate.tensorloglikelihood)))
    end
    if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
      push!(update, :($(job).pstate.tensorlogprior = copy($(job).sstate.pstate.tensorlogprior)))
    end
    push!(update, :($(job).sstate.oldinvtensor = copy($(job).sstate.newinvtensor)))
    push!(update, :($(job).sstate.oldfirstterm = copy($(job).sstate.newfirstterm)))
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

  @gensym iterate_smmala

  result = quote
    function $iterate_smmala()
      $(body...)
    end
  end

  result
end
