function codegen_iterate_ram(job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in MH code generation")
  end

  if job.tuner.verbose
    push!(body, :($(job).sstate.tune.proposed += 1))
  end

  if vform == Univariate
    push!(body, :($(job).sstate.randnsample = randn()))
  elseif vform == Multivariate
    push!(body, :($(job).sstate.randnsample = randn($(job).pstate.size)))
  end

  push!(body, :($(job).sstate.pstate.value = $(job).pstate.value+$(job).sstate.S*$(job).sstate.randnsample))
  push!(body, :($(job).parameter.logtarget!($(job).sstate.pstate)))

  push!(body, :($(job).sstate.ratio = $(job).sstate.pstate.logtarget-$(job).pstate.logtarget))

  if vform == Univariate
    push!(update, :($(job).pstate.value = $(job).sstate.pstate.value))
  elseif vform == Multivariate
    push!(update, :($(job).pstate.value = copy($(job).sstate.pstate.value)))
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

  if job.tuner.verbose
    fmt_iter = format_iteration(ndigits(job.range.burnin))
    fmt_perc = format_percentage()

    push!(body, :(
      if $(job).sstate.tune.totproposed <= $(job).range.burnin && mod($(job).sstate.tune.proposed, $(job).tuner.period) == 0
        rate!($(job).sstate.tune)
        println(
          "Burnin iteration ",
          $(fmt_iter)($(job).sstate.tune.totproposed),
          " of ",
          $(job).range.burnin,
          ": ",
          $(fmt_perc)(100*$(job).sstate.tune.rate),
          " % acceptance rate"
        )
        reset_burnin!($(job).sstate.tune)
      end
    ))
  end

  if vform == Univariate
    push!(body, :($(job).sstate.η = min(1, $(job).count^(-$(job).sampler.γ))))

    push!(body, :($(job).sstate.SST = $(job).sstate.η*(min(1, exp($(job).sstate.ratio))-$(job).sampler.targetrate)))

    push!(body, :($(job).sstate.SST = abs2($(job).sstate.S)*(1+$(job).sstate.SST)))

    push!(body, :($(job).sstate.S = chol($(job).sstate.SST)))
  elseif vform == Multivariate
    push!(body, :($(job).sstate.η = min(1, $(job).pstate.size*$(job).count^(-$(job).sampler.γ))))

    push!(
      body,
      :(
        $(job).sstate.SST = (
          $(job).sstate.randnsample*$(job).sstate.randnsample'/dot($(job).sstate.randnsample, $(job).sstate.randnsample)*
          $(job).sstate.η*(min(1, exp($(job).sstate.ratio))-$(job).sampler.targetrate)
        )
      )
    )

    push!(body, :($(job).sstate.SST = $(job).sstate.S*(eye($(job).pstate.size)+$(job).sstate.SST)*$(job).sstate.S'))

    push!(body, :($(job).sstate.S = chol($(job).sstate.SST)'))
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym iterate_ram

  result = quote
    function $iterate_ram()
      $(body...)
    end
  end

  result
end
