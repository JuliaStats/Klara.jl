### BasicContMuvParameter

mutable struct BasicContMuvParameter <: Parameter{Continuous, Multivariate}
  key::Symbol
  index::Integer
  pdf::Union{ContinuousMultivariateDistribution, Void}
  prior::Union{ContinuousMultivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  gradloglikelihood!::Union{Function, Void}
  gradlogprior!::Union{Function, Void}
  gradlogtarget!::Union{Function, Void}
  tensorloglikelihood!::Union{Function, Void}
  tensorlogprior!::Union{Function, Void}
  tensorlogtarget!::Union{Function, Void}
  dtensorloglikelihood!::Union{Function, Void}
  dtensorlogprior!::Union{Function, Void}
  dtensorlogtarget!::Union{Function, Void}
  uptogradlogtarget!::Union{Function, Void}
  uptotensorlogtarget!::Union{Function, Void}
  uptodtensorlogtarget!::Union{Function, Void}
  diffmethods::Union{DiffMethods, Void}
  diffopts::Union{DiffOptions, Void}
  states::VariableStateVector
  state::ParameterState{Continuous, Multivariate}

  function BasicContMuvParameter(
    key::Symbol,
    index::Integer,
    pdf::Union{ContinuousMultivariateDistribution, Void},
    prior::Union{ContinuousMultivariateDistribution, Void},
    setpdf::Union{Function, Void},
    setprior::Union{Function, Void},
    ll::Union{Function, Void},
    lp::Union{Function, Void},
    lt::Union{Function, Void},
    gll::Union{Function, Void},
    glp::Union{Function, Void},
    glt::Union{Function, Void},
    tll::Union{Function, Void},
    tlp::Union{Function, Void},
    tlt::Union{Function, Void},
    dtll::Union{Function, Void},
    dtlp::Union{Function, Void},
    dtlt::Union{Function, Void},
    uptoglt::Union{Function, Void},
    uptotlt::Union{Function, Void},
    uptodtlt::Union{Function, Void},
    diffmethods::Union{DiffMethods, Void},
    diffopts::Union{DiffOptions, Void},
    states::VariableStateVector,
    state::ParameterState{Continuous, Multivariate}
  )
    args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
    fnames = fieldnames(BasicContMuvParameter)[5:21]

    # Check that all generic functions have correct signature
    for i in 1:17
      if isa(args[i], Function) &&
        isa(args[i], Function) &&
        !method_exists(args[i], (BasicContMuvParameterState, VariableStateVector))
        error("$(fnames[i]) has wrong signature")
      end
    end

    new(
      key,
      index,
      pdf,
      prior,
      setpdf,
      setprior,
      ll,
      lp,
      lt,
      gll,
      glp,
      glt,
      tll,
      tlp,
      tlt,
      dtll,
      dtlp,
      dtlt,
      uptoglt,
      uptotlt,
      uptodtlt,
      diffmethods,
      diffopts,
      states,
      state
    )
  end
end

function BasicContMuvParameter!(
  parameter::BasicContMuvParameter,
  setpdf::Union{Function, Void},
  setprior::Union{Function, Void},
  ll::Union{Function, Void},
  lp::Union{Function, Void},
  lt::Union{Function, Void},
  gll::Union{Function, Void},
  glp::Union{Function, Void},
  glt::Union{Function, Void},
  tll::Union{Function, Void},
  tlp::Union{Function, Void},
  tlt::Union{Function, Void},
  dtll::Union{Function, Void},
  dtlp::Union{Function, Void},
  dtlt::Union{Function, Void},
  uptoglt::Union{Function, Void},
  uptotlt::Union{Function, Void},
  uptodtlt::Union{Function, Void}
)
  args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)

  # Define setpdf (i = 1) and setprior (i = 2)
  for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
    setfield!(
      parameter,
      setter,
      if isa(args[i], Function)
        (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          setfield!(parameter, distribution, args[i](_state, _states))
      else
        nothing
      end
    )
  end

  # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
  # plfield stands for parameter likelihood-related field respectively
  for (i, plfield) in ((3, :loglikelihood!), (6, :gradloglikelihood!))
    setfield!(
      parameter,
      plfield,
      if isa(args[i], Function)
        _state::BasicContMuvParameterState -> args[i](_state, parameter.states)
      else
        nothing
      end
    )
  end

  # Define logprior! (i = 4) and gradlogprior! (i = 7)
  # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
  for (i , ppfield, spfield, f) in ((4, :logprior!, :logprior, logpdf), (7, :gradlogprior!, :gradlogprior, gradlogpdf))
    setfield!(
      parameter,
      ppfield,
      if isa(args[i], Function)
        _state::BasicContMuvParameterState -> args[i](_state, parameter.states)
      else
        if (
            isa(parameter.prior, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(parameter.prior), Vector{eltype(parameter.prior)}))
          ) ||
          isa(args[2], Function)
          _state::BasicContMuvParameterState -> setfield!(_state, spfield, f(getfield(parameter, :prior), _state.value))
        else
          nothing
        end
      end
    )
  end

  # Define logtarget! (i = 5) and gradlogtarget! (i = 8)
  # ptfield, plfield and ppfield stand for parameter target, likelihood and prior-related field respectively
  # stfield, slfield and spfield stand for state target, likelihood and prior-related field respectively
  for (i , ptfield, plfield, ppfield, stfield, slfield, spfield, f) in (
    (5, :logtarget!, :loglikelihood!, :logprior!, :logtarget, :loglikelihood, :logprior, logpdf),
    (8, :gradlogtarget!, :gradloglikelihood!, :gradlogprior!, :gradlogtarget, :gradloglikelihood, :gradlogprior, gradlogpdf)
  )
    setfield!(
      parameter,
      ptfield,
      if isa(args[i], Function)
        _state::BasicContMuvParameterState -> args[i](_state, parameter.states)
      else
        if isa(args[i-2], Function) && isa(getfield(parameter, ppfield), Function)
          function (_state::BasicContMuvParameterState)
            getfield(parameter, plfield)(_state)
            getfield(parameter, ppfield)(_state)
            setfield!(_state, stfield, getfield(_state, slfield)+getfield(_state, spfield))
          end
        elseif (
            isa(parameter.pdf, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(parameter.pdf), Vector{eltype(parameter.pdf)}))
          ) ||
          isa(args[1], Function)
          _state::BasicContMuvParameterState -> setfield!(_state, stfield, f(getfield(parameter, :pdf), _state.value))
        else
          nothing
        end
      end
    )
  end

  # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
  # plfield stands for parameter likelihood-related field respectively
  for (i, plfield) in ((9, :tensorloglikelihood!), (12, :dtensorloglikelihood!))
    setfield!(
      parameter,
      plfield,
      if isa(args[i], Function)
        _state::BasicContMuvParameterState -> args[i](_state, parameter.states)
      else
        nothing
      end
    )
  end

  # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
  # ppfield stands for parameter prior-related field respectively
  for (i, ppfield) in ((10, :tensorlogprior!), (13, :dtensorlogprior!))
    setfield!(
      parameter,
      ppfield,
      if isa(args[i], Function)
        _state::BasicContMuvParameterState -> args[i](_state, parameter.states)
      else
        nothing
      end
    )
  end

  # Define tensorlogtarget! (i = 11) and dtensorlogtarget! (i = 14)
  for (i , ptfield, plfield, ppfield, stfield, slfield, spfield) in (
    (
      11,
      :tensorlogtarget!, :tensorloglikelihood!, :tensorlogprior!,
      :tensorlogtarget, :tensorloglikelihood, :tensorlogprior
    ),
    (
      14,
      :dtensorlogtarget!, :dtensorloglikelihood!, :dtensorlogprior!,
      :dtensorlogtarget, :dtensorloglikelihood, :dtensorlogprior
    )
  )
    setfield!(
      parameter,
      ptfield,
      if isa(args[i], Function)
        _state::BasicContMuvParameterState -> args[i](_state, parameter.states)
      else
        if isa(args[i-2], Function) && isa(args[i-1], Function)
          function (_state::BasicContMuvParameterState)
            getfield(parameter, plfield)(_state)
            getfield(parameter, ppfield)(_state)
            setfield!(_state, stfield, getfield(_state, slfield)+getfield(_state, spfield))
          end
        else
          nothing
        end
      end
    )
  end

  # Define uptogradlogtarget!
  setfield!(
    parameter,
    :uptogradlogtarget!,
    if isa(args[15], Function)
      _state::BasicContMuvParameterState -> args[15](_state, parameter.states)
    else
      if isa(parameter.logtarget!, Function) && isa(parameter.gradlogtarget!, Function)
        function (state::BasicContMuvParameterState)
          parameter.logtarget!(state)
          parameter.gradlogtarget!(state)
        end
      else
        nothing
      end
    end
  )

  # Define uptotensorlogtarget!
  setfield!(
    parameter,
    :uptotensorlogtarget!,
    if isa(args[16], Function)
      _state::BasicContMuvParameterState -> args[16](_state, parameter.states)
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function)
        function (state::BasicContMuvParameterState)
          parameter.logtarget!(state)
          parameter.gradlogtarget!(state)
          parameter.tensorlogtarget!(state)
        end
      else
        nothing
      end
    end
  )

  # Define uptodtensorlogtarget!
  setfield!(
    parameter,
    :uptodtensorlogtarget!,
    if isa(args[17], Function)
      _state::BasicContMuvParameterState -> args[17](_state, parameter.states)
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function) &&
        isa(parameter.dtensorlogtarget!, Function)
        function (state::BasicContMuvParameterState)
          parameter.logtarget!(state)
          parameter.gradlogtarget!(state)
          parameter.tensorlogtarget!(state)
          parameter.dtensorlogtarget!(state)
        end
      else
        nothing
      end
    end
  )
end

BasicContMuvParameter(key::Symbol, index::Integer=0; signature::Symbol=:high, args...) =
  BasicContMuvParameter(key, Val{signature}, index; args...)

function BasicContMuvParameter(
  key::Symbol,
  ::Type{Val{:low}},
  index::Integer=0;
  pdf::Union{ContinuousMultivariateDistribution, Void}=nothing,
  prior::Union{ContinuousMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  gradloglikelihood::Union{Function, Void}=nothing,
  gradlogprior::Union{Function, Void}=nothing,
  gradlogtarget::Union{Function, Void}=nothing,
  tensorloglikelihood::Union{Function, Void}=nothing,
  tensorlogprior::Union{Function, Void}=nothing,
  tensorlogtarget::Union{Function, Void}=nothing,
  dtensorloglikelihood::Union{Function, Void}=nothing,
  dtensorlogprior::Union{Function, Void}=nothing,
  dtensorlogtarget::Union{Function, Void}=nothing,
  uptogradlogtarget::Union{Function, Void}=nothing,
  uptotensorlogtarget::Union{Function, Void}=nothing,
  uptodtensorlogtarget::Union{Function, Void}=nothing,
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  states::VariableStateVector=VariableState[],
  state::ParameterState{Continuous, Multivariate}=BasicContMuvParameterState(0)
)
  parameter = BasicContMuvParameter(key, index, pdf, prior, fill(nothing, 17)..., diffmethods, diffopts, states, state)

  BasicContMuvParameter!(
    parameter,
    setpdf,
    setprior,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorloglikelihood,
    tensorlogprior,
    tensorlogtarget,
    dtensorloglikelihood,
    dtensorlogprior,
    dtensorlogtarget,
    uptogradlogtarget,
    uptotensorlogtarget,
    uptodtensorlogtarget
  )

  parameter
end

function BasicContMuvParameter(
  key::Symbol,
  ::Type{Val{:high}},
  index::Integer=0;
  pdf::Union{ContinuousMultivariateDistribution, Void}=nothing,
  prior::Union{ContinuousMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  gradloglikelihood::Union{Function, Void}=nothing,
  gradlogprior::Union{Function, Void}=nothing,
  gradlogtarget::Union{Function, Void}=nothing,
  tensorloglikelihood::Union{Function, Void}=nothing,
  tensorlogprior::Union{Function, Void}=nothing,
  tensorlogtarget::Union{Function, Void}=nothing,
  dtensorloglikelihood::Union{Function, Void}=nothing,
  dtensorlogprior::Union{Function, Void}=nothing,
  dtensorlogtarget::Union{Function, Void}=nothing,
  uptogradlogtarget::Union{Function, Void}=nothing,
  uptotensorlogtarget::Union{Function, Void}=nothing,
  uptodtensorlogtarget::Union{Function, Void}=nothing,
  nkeys::Integer=0,
  vfarg::Bool=false,
  diffopts::Union{DiffOptions, Void}=nothing,
  states::VariableStateVector=VariableState[],
  state::ParameterState{Continuous, Multivariate}=BasicContMuvParameterState(0)
)
  inargs = (
    setpdf,
    setprior,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorloglikelihood,
    tensorlogprior,
    tensorlogtarget,
    dtensorloglikelihood,
    dtensorlogprior,
    dtensorlogtarget,
    uptogradlogtarget,
    uptotensorlogtarget,
    uptodtensorlogtarget
  )

  fnames = fieldnames(BasicContMuvParameterState)[2:13]

  if vfarg
    if nkeys > 0
      if diffopts != nothing
        error("In the case of autodiff, if nkeys is not 0, then vfarg must be false")
      end
    else
      error("If vfarg=$vfarg, nkeys must be positive, got $nkeys")
    end
  else
    @assert nkeys >= 0 "If vfarg=$vfarg, nkeys must be non-negative, got $nkeys"
  end

  if diffopts != nothing
    for i in 3:5
      if isa(inargs[i], Function) && (!isa(inargs[i+3], Function) || (diffopts.order == 2 && !isa(inargs[i+6], Function)))
        diffopts.targets[i-2] = true
      end
    end
  end

  parameter = BasicContMuvParameter(key, index, pdf, prior, fill(nothing, 17)..., DiffMethods(), diffopts, states, state)

  outargs = Union{Function, Void}[nothing for i in 1:17]

  if vfarg
    for i in 1:2
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          inargs[i](Any[s.value for s in _states])
      end
    end

    for i in 3:14
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          setfield!(_state, fnames[i-2], inargs[i](Any[s.value for s in _states]))
      end
    end

    if isa(inargs[15], Function)
      outargs[15] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
        (_state.logtarget, _state.gradlogtarget) = inargs[15](Any[s.value for s in _states])
    end

    if isa(inargs[16], Function)
      outargs[16] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
        (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) = inargs[16](Any[s.value for s in _states])
    end

    if isa(inargs[17], Function)
      outargs[17] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
        (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget, _state.dtensorlogtarget) =
          inargs[17](Any[s.value for s in _states])
    end
  else
    if nkeys > 0
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
            inargs[i](_state.value, Any[s.value for s in _states])
        end
      end

      for i in 3:14
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value, Any[s.value for s in _states]))
        end
      end

      if isa(inargs[15], Function)
        outargs[15] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget) = inargs[15](_state.value, Any[s.value for s in _states])
      end

      if isa(inargs[16], Function)
        outargs[16] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) =
            inargs[16](_state.value, Any[s.value for s in _states])
      end

      if isa(inargs[17], Function)
        outargs[17] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget, _state.dtensorlogtarget) =
            inargs[17](_state.value, Any[s.value for s in _states])
      end
    else
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) -> inargs[i](_state.value)
        end
      end

      for i in 3:14
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value))
        end
      end

      if isa(inargs[15], Function)
        outargs[15] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget) = inargs[15](_state.value)
      end

      if isa(inargs[16], Function)
        outargs[16] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) = inargs[16](_state.value)
      end

      if isa(inargs[17], Function)
        outargs[17] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget, _state.dtensorlogtarget) =
            inargs[17](_state.value)
      end
    end
  end

  for (i, j, field, distribution, f) in ((4, 2, :logprior, :prior, logpdf), (5, 1, :logtarget, :pdf, logpdf))
    if !isa(inargs[i], Function) &&
      (
        (
          isa(getfield(parameter, distribution), ContinuousMultivariateDistribution) &&
          method_exists(f, (typeof(getfield(parameter, distribution)), Vector{eltype(getfield(parameter, distribution))}))
        ) ||
        isa(inargs[j], Function)
      )
      outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
        setfield!(_state, field, f(getfield(parameter, distribution), _state.value))
    end
  end

  if diffopts != nothing
    if diffopts.mode == :reverse
      for (i, field) in ((3, :closurell), (4, :closurelp), (5, :closurelt))
        if isa(inargs[i], Function)
          setfield!(
            parameter.diffmethods,
            field,
            nkeys == 0 ? inargs[i] : _x -> inargs[i](_x, Any[s.value for s in parameter.states])
          )
        end
      end

      for (i, returnname, diffresult, diffmethod) in (
        (6, :gradloglikelihood, :resultll, :tapegll),
        (7, :gradlogprior, :resultlp, :tapeglp),
        (8, :gradlogtarget, :resultlt, :tapetlt)
      )
        if !isa(inargs[i], Function) && isa(inargs[i-3], Function)
          outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
            setfield!(
              _state,
              returnname,
              reverse_autodiff_gradient(
                getfield(_state.diffstate, diffresult), getfield(_state.diffmethods, diffmethod), _state.value
              )
            )
        end
      end

      if !isa(inargs[15], Function) && isa(inargs[5], Function)
        outargs[15] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget) =
            reverse_autodiff_upto_gradient(_state.diffstate.resultlt, _state.diffmethods.tapeglt, _state.value)
      end

      if diffopts.order == 2
        for (i, returnname, diffresult, diffmethod) in (
          (9, :tensorloglikelihood, :resultll, :tapetll),
          (10, :tensorlogprior, :resultlp, :tapetlp),
          (11, :tensorlogtarget, :resultlt, :tapetlt)
        )
          if !isa(inargs[i], Function) && isa(inargs[i-6], Function)
            outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
              setfield!(
                _state,
                returnname,
                reverse_autodiff_negative_hessian(
                  getfield(_state.diffstate, diffresult),
                  getfield(_state.diffmethods, diffmethod),
                  _state.value
                )
              )
          end
        end

        if !isa(inargs[16], Function) && isa(inargs[5], Function)
          outargs[16] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
            (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) =
              reverse_autodiff_upto_negative_hessian(_state.diffstate.resultlt, _state.diffmethods.tapetlt, _state.value)
        end
      end
    elseif diffopts.mode == :forward
      for (i, field) in ((3, :closurell), (4, :closurelp), (5, :closurelt))
        if isa(inargs[i], Function)
          setfield!(
            parameter.diffmethods,
            field,
            nkeys == 0 ? inargs[i] : _x::Vector -> inargs[i](_x, Any[s.value for s in parameter.states])
          )
        end
      end

      for (i, returnname, diffresult, diffmethod, diffconfig) in (
        (6, :gradloglikelihood, :resultll, :closurell, :cfggll),
        (7, :gradlogprior, :resultlp, :closurelp, :cfgglp),
        (8, :gradlogtarget, :resultlt, :closurelt, :cfgglt)
      )
        if !isa(inargs[i], Function) && isa(inargs[i-3], Function)
          outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
            setfield!(
              _state,
              returnname,
              forward_autodiff_gradient(
                getfield(_state.diffstate, diffresult),
                getfield(_state.diffmethods, diffmethod),
                _state.value,
                getfield(_state.diffstate, diffconfig)
              )
            )
        end
      end

      if !isa(inargs[15], Function) && isa(inargs[5], Function)
        outargs[15] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget) =
            forward_autodiff_upto_gradient(
              _state.diffstate.resultlt, _state.diffmethods.closurelt, _state.value, _state.diffstate.cfgglt
            )
      end

      if diffopts.order == 2
        for (i, returnname, diffresult, diffmethod, diffconfig) in (
          (9, :tensorloglikelihood, :resultll, :closurell, :cfgtll),
          (10, :tensorlogprior, :resultlp, :closurelp, :cfgtlp),
          (11, :tensorlogtarget, :resultlt, :closurelt, :cfgtlt)
        )
          if !isa(inargs[i], Function) && isa(inargs[i-6], Function)
            outargs[i] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
              setfield!(
                _state,
                returnname,
                forward_autodiff_negative_hessian(
                  getfield(_state.diffstate, diffresult),
                  getfield(_state.diffmethods, diffmethod),
                  _state.value,
                  getfield(_state.diffstate, diffconfig)
                )
              )
          end
        end

        if !isa(inargs[16], Function) && isa(inargs[5], Function)
          outargs[16] = (_state::BasicContMuvParameterState, _states::VariableStateVector) ->
            (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) =
              forward_autodiff_upto_negative_hessian(
                _state.diffstate.resultlt, _state.diffmethods.closurelt, _state.value, _state.diffstate.cfgtlt
              )
        end
      end
    end
  end

  BasicContMuvParameter!(parameter, outargs...)

  parameter
end

function set_tapes!(parameter::BasicContMuvParameter, pstate::ParameterState{Continuous, Multivariate})
  diffgtapes = (:tapegll, :tapeglp, :tapeglt)
  diffttapes = (:tapetll, :tapetlp, :tapetlt)

  for (i, diffclosure) in ((1, :closurell), (2, :closurelp), (3, :closurelt))
    if parameter.diffopts.targets[i]
      setfield!(
        parameter.diffmethods,
        diffgtapes[i],
        ReverseDiff.GradientTape(getfield(parameter.diffmethods, diffclosure), pstate.value)
      )

      if parameter.diffopts.compiled
        setfield!(
          parameter.diffmethods,
          diffgtapes[i],
          ReverseDiff.compile(getfield(parameter.diffmethods, diffgtapes[i]))
        )
      end

      if parameter.diffopts.order == 2
        difftapes = (:tapegll, :tapeglp, :tapeglt)

        setfield!(
          parameter.diffmethods,
          diffttapes[i],
          ReverseDiff.HessianTape(getfield(parameter.diffmethods, diffclosure), pstate.value)
        )

        if parameter.diffopts.compiled
          setfield!(
            parameter.diffmethods,
            diffttapes[i],
            ReverseDiff.compile(getfield(parameter.diffmethods, diffttapes[i]))
          )
        end
      end
    end
  end
end

value_support(::Type{BasicContMuvParameter}) = Continuous
value_support(::BasicContMuvParameter) = Continuous

variate_form(::Type{BasicContMuvParameter}) = Multivariate
variate_form(::BasicContMuvParameter) = Multivariate

default_state_type(::BasicContMuvParameter) = BasicContMuvParameterState

default_state(variable::BasicContMuvParameter, value::Vector{N}, outopts::Dict) where {N<:Real} =
  BasicContMuvParameterState(
    value,
    [getfield(variable, fieldnames(BasicContMuvParameter)[i]) == nothing ? false : true for i in 10:18],
    (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[],
    variable.diffmethods,
    variable.diffopts
  )

show(io::IO, ::Type{BasicContMuvParameter}) = print(io, "BasicContMuvParameter")
