function codegen(::Type{Val{:iterate}}, job::BasicMCJob, ::Type{AM})
  local result::Expr
  update = []
  noupdate = []
  body = []
  local dindex::Integer

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in AM code generation")
  end

  push!(body, :(_job.sstate.count += 1))

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  if vform == Univariate
    push!(
      body,
      :(
        if _job.sstate.count <= _job.sampler.t0
          set_normal!(_job.sstate, _job.sampler, _job.pstate)
        else
          _job.sstate.C = covariance(
            _job.sstate.C, _job.sstate.count-2, _job.pstate.value, _job.sstate.lastmean, _job.sstate.secondlastmean
          )

          set_gmm!(_job.sstate, _job.sampler, _job.pstate)
        end
      )
    )
  elseif vform == Multivariate
    push!(
      body,
      :(
        if _job.sstate.count <= _job.sampler.t0
          set_normal!(_job.sstate, _job.sampler, _job.pstate)
        else
          covariance!(
            _job.sstate.C,
            _job.sstate.C,
            _job.sstate.count-2,
            _job.pstate.value,
            _job.sstate.lastmean,
            _job.sstate.secondlastmean
          )

          _job.sstate.C[:, :] = Hermitian(_job.sstate.C)

          set_gmm!(_job.sstate, _job.sampler, _job.pstate)
        end
      )
    )
  end

  if vform == Univariate
    push!(body, :(_job.sstate.pstate.value = rand(_job.sstate.proposal)))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.pstate.value[:] = rand(_job.sstate.proposal)))
  end

  push!(body, :(_job.parameter.logtarget!(_job.sstate.pstate)))

  push!(body, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(
    body,
    :(
      if _job.sstate.count > _job.sampler.t0
        _job.sstate.ratio -= logpdf(_job.sstate.proposal, _job.sstate.pstate.value)
        set_gmm!(_job.sstate, _job.sampler, _job.sstate.pstate)
        _job.sstate.ratio += logpdf(_job.sstate.proposal, _job.pstate.value)
      end
    )
  )

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

  if vform == Univariate
    push!(body, :(_job.sstate.secondlastmean = _job.sstate.lastmean))
    push!(body, :(_job.sstate.lastmean = recursive_mean(_job.sstate.lastmean, _job.sstate.count, _job.pstate.value)))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.secondlastmean = copy(_job.sstate.lastmean)))
    push!(body, :(recursive_mean!(_job.sstate.lastmean, _job.sstate.lastmean, _job.sstate.count, _job.pstate.value)))
  end

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

  @gensym _iterate

  result = quote
    function $_iterate(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end
