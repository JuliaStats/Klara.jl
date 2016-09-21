function codegen(::Type{Val{:iterate}}, ::Type{NUTS}, job::BasicMCJob)
  result::Expr
  whilebody = []
  update = []
  noupdate = []
  burninbody = []
  ifburninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in NUTS code generation")
  end

  # push!(body, :(_job.sstate.count += 1))

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

  push!(body, :(_job.sstate.u = log(rand())+_job.sstate.oldhamiltonian))

  push!(whilebody, :(_job.sstate.v = rand(Bool) ? 1 : -1))

  push!(whilebody, :(
    if _job.sstate.v == -1
      _job.sstate.pstateminus,
      _job.sstate.momentumminus,
      _,
      _,
      _job.sstate.pstateprime,
      _job.sstate.nprime,
      _job.sstate.sprime =
        build_tree!(
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
        build_tree!(
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

  if vform == Univariate
    push!(update, :(_job.pstate.value = _job.sstate.pstateprime.value))
    push!(update, :(_job.pstate.gradlogtarget = _job.sstate.pstateprime.gradlogtarget))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :(_job.pstate.gradloglikelihood = _job.sstate.pstateprime.gradloglikelihood))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :(_job.pstate.gradlogprior = _job.sstate.pstateprime.gradlogprior))
    end
  elseif vform == Multivariate
    push!(update, :(_job.pstate.value = copy(_job.sstate.pstateprime.value)))
    push!(update, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstateprime.gradlogtarget)))
    if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
      push!(update, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstateprime.gradloglikelihood)))
    end
    if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
      push!(update, :(_job.pstate.gradlogprior = copy(_job.sstate.pstateprime.gradlogprior)))
    end
  end
  push!(update, :(_job.pstate.logtarget = _job.sstate.pstateprime.logtarget))
  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstateprime.loglikelihood))
  end
  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstateprime.logprior))
  end
  push!(update, :(_job.sstate.count += 1))

  push!(
    whilebody,
    Expr(:if, :(_job.sstate.sprime && (rand() < _job.sstate.nprime/_job.sstate.n)), Expr(:block, update...))
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
