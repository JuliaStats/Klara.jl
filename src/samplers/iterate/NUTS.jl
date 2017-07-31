function codegen(::Type{Val{:iterate}}, ::Type{NUTS}, job::BasicMCJob)
  local result::Expr
  whilebody = []
  ifwhilebody = []
  burninbody = []
  ifburninbody = []
  body = []
  local dindex::Integer

  if isa(job.tuner, DualAveragingMCTuner)
    push!(body, :(local a::Real))
    push!(body, :(local na::Integer))
  end

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in NUTS code generation")
  end

  if isa(job.tuner, DualAveragingMCTuner)
    push!(body, :(_job.sstate.count += 1))
  end

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  if vform == Univariate
    push!(body, :(_job.sstate.momentum = randn()))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.momentum[:] = randn(_job.pstate.size)))
  end

  push!(body, :(_job.sstate.oldhamiltonian = hamiltonian(_job.pstate.logtarget, _job.sstate.momentum)))

  if vform == Univariate
    push!(body, :(_job.sstate.pstateplus.value = _job.pstate.value))
    push!(body, :(_job.sstate.pstateplus.gradlogtarget = _job.pstate.gradlogtarget))
    push!(body, :(_job.sstate.momentumplus = _job.sstate.momentum))
    push!(body, :(_job.sstate.pstateminus.value = _job.pstate.value))
    push!(body, :(_job.sstate.pstateminus.gradlogtarget = _job.pstate.gradlogtarget))
    push!(body, :(_job.sstate.momentumminus = _job.sstate.momentum))
  elseif vform == Multivariate
    push!(body, :(_job.sstate.pstateplus.value = copy(_job.pstate.value)))
    push!(body, :(_job.sstate.pstateplus.gradlogtarget = copy(_job.pstate.gradlogtarget)))
    push!(body, :(_job.sstate.momentumplus = copy(_job.sstate.momentum)))
    push!(body, :(_job.sstate.pstateminus.value = copy(_job.pstate.value)))
    push!(body, :(_job.sstate.pstateminus.gradlogtarget = copy(_job.pstate.gradlogtarget)))
    push!(body, :(_job.sstate.momentumminus = copy(_job.sstate.momentum)))
  end

  push!(body, :(_job.sstate.j = 0))
  push!(body, :(_job.sstate.n = 1))
  push!(body, :(_job.sstate.s = true))
  if in(:accept, job.outopts[:diagnostics]) || job.tuner.verbose
    push!(body, :(_job.sstate.update = false))
  end

  push!(body, :(_job.sstate.u = log(rand())+_job.sstate.oldhamiltonian))

  push!(whilebody, :(_job.sstate.v = rand(Bool) ? 1 : -1))

  if isa(job.tuner, VanillaMCTuner)
    push!(whilebody, :(
      if _job.sstate.v == -1
        _job.sstate.pstateminus,
        _job.sstate.momentumminus,
        _,
        _,
        _job.sstate.pstateprime,
        _job.sstate.nprime,
        _job.sstate.sprime =
          _job.sampler.buildtree!(
            _job.sstate,
            _job.sstate.pstateminus,
            _job.sstate.momentumminus,
            _job.sstate.u,
            _job.sstate.v,
            _job.sstate.j,
            _job.sstate.tune.step,
            _job.parameter.logtarget!,
            _job.parameter.gradlogtarget!,
            _job.sampler
          )
      else
        _,
        _,
        _job.sstate.pstateplus,
        _job.sstate.momentumplus,
        _job.sstate.pstateprime,
        _job.sstate.nprime,
        _job.sstate.sprime =
          _job.sampler.buildtree!(
            _job.sstate,
            _job.sstate.pstateplus,
            _job.sstate.momentumplus,
            _job.sstate.u,
            _job.sstate.v,
            _job.sstate.j,
            _job.sstate.tune.step,
            _job.parameter.logtarget!,
            _job.parameter.gradlogtarget!,
            _job.sampler
          )
      end
    ))
  elseif isa(job.tuner, DualAveragingMCTuner)
    push!(whilebody, :(
      if _job.sstate.v == -1
        _job.sstate.pstateminus,
        _job.sstate.momentumminus,
        _,
        _,
        _job.sstate.pstateprime,
        _job.sstate.nprime,
        _job.sstate.sprime,
        a,
        na =
          _job.sampler.buildtree!(
            _job.sstate,
            _job.sstate.pstateminus,
            _job.sstate.momentumminus,
            _job.sstate.oldhamiltonian,
            _job.sstate.u,
            _job.sstate.v,
            _job.sstate.j,
            _job.sstate.tune.step,
            _job.parameter.logtarget!,
            _job.parameter.gradlogtarget!,
            _job.sampler
          )
      else
        _,
        _,
        _job.sstate.pstateplus,
        _job.sstate.momentumplus,
        _job.sstate.pstateprime,
        _job.sstate.nprime,
        _job.sstate.sprime,
        a,
        na =
          _job.sampler.buildtree!(
            _job.sstate,
            _job.sstate.pstateplus,
            _job.sstate.momentumplus,
            _job.sstate.oldhamiltonian,
            _job.sstate.u,
            _job.sstate.v,
            _job.sstate.j,
            _job.sstate.tune.step,
            _job.parameter.logtarget!,
            _job.parameter.gradlogtarget!,
            _job.sampler
          )
      end
    ))
  end

  if vform == Univariate
    push!(ifwhilebody, :(_job.pstate.value = _job.sstate.pstateprime.value))
    push!(ifwhilebody, :(_job.pstate.gradlogtarget = _job.sstate.pstateprime.gradlogtarget))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(ifwhilebody, :(_job.pstate.gradloglikelihood = _job.sstate.pstateprime.gradloglikelihood))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(ifwhilebody, :(_job.pstate.gradlogprior = _job.sstate.pstateprime.gradlogprior))
    end
  elseif vform == Multivariate
    push!(ifwhilebody, :(_job.pstate.value = copy(_job.sstate.pstateprime.value)))
    push!(ifwhilebody, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstateprime.gradlogtarget)))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(ifwhilebody, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstateprime.gradloglikelihood)))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(ifwhilebody, :(_job.pstate.gradlogprior = copy(_job.sstate.pstateprime.gradlogprior)))
    end
  end
  push!(ifwhilebody, :(_job.pstate.logtarget = _job.sstate.pstateprime.logtarget))
  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(ifwhilebody, :(_job.pstate.loglikelihood = _job.sstate.pstateprime.loglikelihood))
  end
  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(ifwhilebody, :(_job.pstate.logprior = _job.sstate.pstateprime.logprior))
  end
  if in(:accept, job.outopts[:diagnostics]) || job.tuner.verbose
    push!(ifwhilebody, :(_job.sstate.update = true))
  end

  push!(
    whilebody,
    Expr(:if, :(_job.sstate.sprime && (rand() < _job.sstate.nprime/_job.sstate.n)), Expr(:block, ifwhilebody...))
  )

  push!(whilebody, :(_job.sstate.j += 1))
  push!(whilebody, :(_job.sstate.n += _job.sstate.nprime))
  push!(
    whilebody,
    :(
      _job.sstate.s =
      _job.sstate.sprime &&
      !uturn(
        _job.sstate.pstateplus.value,
        _job.sstate.pstateminus.value,
        _job.sstate.momentumplus,
        _job.sstate.momentumminus
      )
    )
  )

  push!(body, Expr(:while, :(_job.sstate.s && (_job.sstate.j < _job.sampler.maxndoublings)), Expr(:block, whilebody...)))

  dindex = findfirst(job.outopts[:diagnostics], :accept)
  if dindex != 0
    push!(body, :( _job.pstate.diagnosticvalues[$dindex] = _job.sstate.update))
  end

  dindex = findfirst(job.outopts[:diagnostics], :ndoublings)
  if dindex != 0
    push!(body, :( _job.pstate.diagnosticvalues[$dindex] = _job.sstate.j))
  end

  dindex = findfirst(job.outopts[:diagnostics], :a)
  if dindex != 0
    push!(body, :( _job.pstate.diagnosticvalues[$dindex] = a))
  end

  dindex = findfirst(job.outopts[:diagnostics], :na)
  if dindex != 0
    push!(body, :( _job.pstate.diagnosticvalues[$dindex] = na))
  end

  if job.tuner.verbose
    push!(body, :(_job.sstate.update && (_job.sstate.tune.accepted += 1)))
  end

  if isa(job.tuner, VanillaMCTuner) && job.tuner.verbose
    push!(burninbody, :(rate!(_job.sstate.tune)))

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
    push!(burninbody, :(tune!(_job.sstate.tune, _job.tuner, _job.sstate.count, a/na)))

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

  @gensym _iterate

  result = quote
    function $_iterate(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end
