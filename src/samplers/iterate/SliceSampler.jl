function codegen(::Type{Val{:iterate}}, ::Type{SliceSampler}, job::BasicMCJob)
  local result::Expr
  innerbody = []
  burninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in SliceSampler code generation")
  end

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  if vform == Multivariate
    push!(innerbody, :(_job.sstate.loguprime = log(rand())+_job.pstate.logtarget))
    push!(innerbody, :(_job.sstate.lstate.value = copy(_job.pstate.value)))
    push!(innerbody, :(_job.sstate.rstate.value = copy(_job.pstate.value)))
    push!(innerbody, :(_job.sstate.primestate.value = copy(_job.pstate.value)))

    push!(innerbody, :(_job.sstate.runiform = rand()))
    push!(innerbody, :(_job.sstate.lstate.value[i] = _job.pstate.value[i]-_job.sstate.runiform*_job.sampler.widths[i]))
    push!(innerbody, :(_job.sstate.rstate.value[i] = _job.pstate.value[i]+(1-_job.sstate.runiform)*_job.sampler.widths[i]))

    if job.sampler.stepout
      push!(innerbody, :(_job.parameter.logtarget!(_job.sstate.lstate)))
      push!(innerbody, :(
        while _job.sstate.lstate.logtarget > _job.sstate.loguprime
          _job.sstate.lstate.value[i] -= _job.sampler.widths[i]
          _job.parameter.logtarget!(_job.sstate.lstate)
        end
      ))

      push!(innerbody, :(_job.parameter.logtarget!(_job.sstate.rstate)))
      push!(innerbody, :(
        while _job.sstate.rstate.logtarget > _job.sstate.loguprime
          _job.sstate.rstate.value[i] += _job.sampler.widths[i]
          _job.parameter.logtarget!(_job.sstate.rstate)
        end
      ))
    end

    push!(innerbody, :(
      while true
        _job.sstate.primestate.value[i] =
          rand()*(_job.sstate.rstate.value[i]-_job.sstate.lstate.value[i])+_job.sstate.lstate.value[i]
        _job.pstate.logtarget = _job.parameter.logtarget!(_job.sstate.primestate)
        if _job.pstate.logtarget > _job.sstate.loguprime
          break
        else
          if _job.sstate.primestate.value[i] > _job.pstate.value[i]
            _job.sstate.rstate.value[i] = _job.sstate.primestate.value[i]
          elseif _job.sstate.primestate.value[i] < _job.pstate.value[i]
            _job.sstate.lstate.value[i] = _job.sstate.primestate.value[i]
          else
            @assert false "Shrunk to current position and still not acceptable"
          end
        end
      end
    ))

    push!(innerbody, :(_job.pstate.value[i] = _job.sstate.primestate.value[i]))

    push!(body, Expr(:for, :(i = 1:_job.pstate.size), Expr(:block, innerbody...)))
  end

  if job.tuner.verbose
    fmt_iter = format_iteration(ndigits(job.range.burnin))

    push!(burninbody, :(println("Burnin iteration ", $(fmt_iter)(_job.sstate.tune.totproposed), " of ", _job.range.burnin)))
  end

  push!(burninbody, :(_job.sstate.tune.totproposed += _job.sstate.tune.proposed))
  push!(burninbody, :(_job.sstate.tune.proposed = 0))

  push!(
    body,
    Expr(
      :if,
      :(_job.sstate.tune.totproposed <= _job.range.burnin && mod(_job.sstate.tune.proposed, _job.tuner.period) == 0),
      Expr(:block, burninbody...)
    )
  )

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
