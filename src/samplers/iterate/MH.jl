function codegen_iterate_mh(job::BasicMCJob, outopts::Dict, plain::Bool)
  result::Expr
  update = []
  noupdate = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in MALA code generation")
  end

  if job.tuner.verbose
    push!(body, :(_sstate.tune.proposed += 1))
  end

  push!(body, :(_sstate.pstate.value = _sampler.randproposal(_pstate.value)))
  push!(body, :(_parameter.logtarget!(_sstate.pstate)))

  if job.sampler.symmetric
    push!(body, :(_sstate.ratio = _sstate.pstate.logtarget-_pstate.logtarget))
  else
    push!(body, :(_sstate.ratio = (
      _sstate.pstate.logtarget
      +_sampler.logproposal(_sstate.pstate.value, _pstate.value)
      -_pstate.logtarget
      -_sampler.logproposal(_pstate.value, _sstate.pstate.value)
    )))
  end

  if vform == Univariate
    push!(update, :(_pstate.value = _sstate.pstate.value))
  elseif vform == Multivariate
    push!(update, :(_pstate.value = copy(_sstate.pstate.value)))
  end
  push!(update, :(_pstate.logtarget = _sstate.pstate.logtarget))
  if in(:loglikelihood, outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_pstate.loglikelihood = _sstate.pstate.loglikelihood))
  end
  if in(:logprior, outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_pstate.logprior = _sstate.pstate.logprior))
  end
  if in(:accept, outopts[:diagnostics])
    push!(update, :(_pstate.diagnosticvalues[1] = true))
    push!(noupdate, :(_pstate.diagnosticvalues[1] = false))
  end
  if job.tuner.verbose
    push!(update, :(_sstate.tune.accepted += 1))
  end

  push!(body, Expr(:if, :(_sstate.ratio > 0 || (_sstate.ratio > log(rand()))), Expr(:block, update...), noupdate...))

  if job.tuner.verbose
    push!(body, :(
      if _sstate.tune.totproposed <= _range.burnin && mod(_sstate.tune.proposed, _tuner.period) == 0
        rate!(_sstate.tune)
        println(
          "Burnin iteration ",
          _sstate.tune.totproposed,
          " of ",
          _range.burnin,
          ": ",
          round(100*_sstate.tune.rate, 2),
          " % acceptance rate"
        )
        reset_burnin!(_sstate.tune)
      end
    ))
  end

  if !plain
    push!(body, :(produce()))
  end

  @gensym iterate_mh

  result = quote
    function $iterate_mh(
      _pstate::$(typeof(job.pstate)),
      _sstate::$(typeof(job.sstate)),
      _parameter::$(typeof(job.parameter)),
      _sampler::MH,
      _tuner::$(typeof(job.tuner)),
      _range::$(typeof(job.range))
    )
      $(body...)
    end
  end

  result
end
