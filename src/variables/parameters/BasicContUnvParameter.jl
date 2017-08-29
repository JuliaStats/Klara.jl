### BasicContUnvParameter

# Guidelines for usage of inner constructors of continuous parameter types:
# 1) Function fields have higher priority than implicitly derived definitions via the pdf field
# 2) Target-related fields have higher priority than implicitly derived likelihood+prior fields
# 3) Upto-related fields have higher priority than implicitly derived Function tuples

### BasicContUnvParameter

mutable struct BasicContUnvParameter <: Parameter{Continuous, Univariate}
  key::Symbol
  index::Integer
  pdf::Union{ContinuousUnivariateDistribution, Void}
  prior::Union{ContinuousUnivariateDistribution, Void}
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
  state::ParameterState{Continuous, Univariate}

  function BasicContUnvParameter(
    key::Symbol,
    index::Integer,
    pdf::Union{ContinuousUnivariateDistribution, Void},
    prior::Union{ContinuousUnivariateDistribution, Void},
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
    state::ParameterState{Continuous, Univariate}
  )
    args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
    fnames = fieldnames(BasicContUnvParameter)[5:21]

    # Check that all generic functions have correct signature
    for i in 1:17
      if isa(args[i], Function) &&
        isa(args[i], Function) &&
        !method_exists(args[i], (BasicContUnvParameterState, VariableStateVector))
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

function BasicContUnvParameter!(
  parameter::BasicContUnvParameter,
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
        (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
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
        _state::BasicContUnvParameterState -> args[i](_state, parameter.states)
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
        _state::BasicContUnvParameterState -> args[i](_state, parameter.states)
      else
        if (
            isa(parameter.prior, ContinuousUnivariateDistribution) &&
            method_exists(f, (typeof(parameter.prior), eltype(parameter.prior)))
          ) ||
          isa(args[2], Function)
          _state::BasicContUnvParameterState -> setfield!(_state, spfield, f(getfield(parameter, :prior), _state.value))
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
        _state::BasicContUnvParameterState -> args[i](_state, parameter.states)
      else
        if isa(args[i-2], Function) && isa(getfield(parameter, ppfield), Function)
          function (_state::BasicContUnvParameterState)
            getfield(parameter, plfield)(_state)
            getfield(parameter, ppfield)(_state)
            setfield!(_state, stfield, getfield(_state, slfield)+getfield(_state, spfield))
          end
        elseif (
            isa(parameter.pdf, ContinuousUnivariateDistribution) &&
            method_exists(f, (typeof(parameter.pdf), eltype(parameter.pdf)))
          ) ||
          isa(args[1], Function)
          _state::BasicContUnvParameterState -> setfield!(_state, stfield, f(getfield(parameter, :pdf), _state.value))
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
        _state::BasicContUnvParameterState -> args[i](_state, parameter.states)
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
        _state::BasicContUnvParameterState -> args[i](_state, parameter.states)
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
        _state::BasicContUnvParameterState -> args[i](_state, parameter.states)
      else
        if isa(args[i-2], Function) && isa(args[i-1], Function)
          function (_state::BasicContUnvParameterState)
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
      _state::BasicContUnvParameterState -> args[15](_state, parameter.states)
    else
      if isa(parameter.logtarget!, Function) && isa(parameter.gradlogtarget!, Function)
        function (state::BasicContUnvParameterState)
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
      _state::BasicContUnvParameterState -> args[16](_state, parameter.states)
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function)
        function (state::BasicContUnvParameterState)
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
      _state::BasicContUnvParameterState -> args[17](_state, parameter.states)
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function) &&
        isa(parameter.dtensorlogtarget!, Function)
        function (state::BasicContUnvParameterState)
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

BasicContUnvParameter(key::Symbol, index::Integer=0; signature::Symbol=:high, args...) =
  BasicContUnvParameter(key, Val{signature}, index; args...)

function BasicContUnvParameter(
  key::Symbol,
  ::Type{Val{:low}},
  index::Integer=0;
  pdf::Union{ContinuousUnivariateDistribution, Void}=nothing,
  prior::Union{ContinuousUnivariateDistribution, Void}=nothing,
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
  state::ParameterState{Continuous, Univariate}=BasicContUnvParameterState()
)
  parameter = BasicContUnvParameter(key, index, pdf, prior, fill(nothing, 17)..., diffmethods, diffopts, states, state)

  BasicContUnvParameter!(
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

function BasicContUnvParameter(
  key::Symbol,
  ::Type{Val{:high}},
  index::Integer=0;
  pdf::Union{ContinuousUnivariateDistribution, Void}=nothing,
  prior::Union{ContinuousUnivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Expr, Void}=nothing,
  logprior::Union{Function, Expr, Void}=nothing,
  logtarget::Union{Function, Expr, Void}=nothing,
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
  state::ParameterState{Continuous, Univariate}=BasicContUnvParameterState()
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

  fnames = fieldnames(BasicContUnvParameterState)[2:13]

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
    if diffopts.mode != :forward
      error("Only :forward mode autodiff is available for univariate parameters, got $autodiff mode")
    end

    if diffopts.order != 1
      error("Derivative order must be 1, got order=$order")
    end
  end

  if diffopts != nothing
    for i in 3:5
      if isa(inargs[i], Function) && !isa(inargs[i+3], Function)
        diffopts.targets[i-2] = true
      end
    end
  end

  parameter = BasicContUnvParameter(key, index, pdf, prior, fill(nothing, 17)..., DiffMethods(), diffopts, states, state)

  outargs = Union{Function, Void}[nothing for i in 1:17]

  if vfarg
    for i in 1:2
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          inargs[i](Any[s.value for s in _states])
      end
    end

    for i in 3:14
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          setfield!(_state, fnames[i-2], inargs[i](Any[s.value for s in _states]))
      end
    end

    if isa(inargs[15], Function)
      outargs[15] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
        (_state.logtarget, _state.gradlogtarget) = inargs[15](Any[s.value for s in _states])
    end

    if isa(inargs[16], Function)
      outargs[16] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
        (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) = inargs[16](Any[s.value for s in _states])
    end

    if isa(inargs[17], Function)
      outargs[17] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
        (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget, _state.dtensorlogtarget) =
          inargs[17](Any[s.value for s in _states])
    end
  else
    if nkeys > 0
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
            inargs[i](_state.value, Any[s.value for s in _states])
        end
      end

      for i in 3:14
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value, Any[s.value for s in _states]))
        end
      end

      if isa(inargs[15], Function)
        outargs[15] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget) = inargs[15](_state.value, Any[s.value for s in _states])
      end

      if isa(inargs[16], Function)
        outargs[16] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) =
            inargs[16](_state.value, Any[s.value for s in _states])
      end

      if isa(inargs[17], Function)
        outargs[17] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget, _state.dtensorlogtarget) =
            inargs[17](_state.value, Any[s.value for s in _states])
      end
    else
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) -> inargs[i](_state.value)
        end
      end

      for i in 3:14
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value))
        end
      end

      if isa(inargs[15], Function)
        outargs[15] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget) = inargs[15](_state.value)
      end

      if isa(inargs[16], Function)
        outargs[16] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget) = inargs[16](_state.value)
      end

      if isa(inargs[17], Function)
        outargs[17] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          (_state.logtarget, _state.gradlogtarget, _state.tensorlogtarget, _state.dtensorlogtarget) =
            inargs[17](_state.value)
      end
    end
  end

  for (i, j, field, distribution, f) in ((4, 2, :logprior, :prior, logpdf), (5, 1, :logtarget, :pdf, logpdf))
    if !isa(inargs[i], Function) &&
      (
        (
          isa(getfield(parameter, distribution), ContinuousUnivariateDistribution) &&
          method_exists(f, (typeof(getfield(parameter, distribution)), eltype(getfield(parameter, distribution))))
        ) ||
        isa(inargs[j], Function)
      )
      outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
        setfield!(_state, field, f(getfield(parameter, distribution), _state.value))
    end
  end

  if diffopts != nothing
    for (i, field) in ((3, :closurell), (4, :closurelp), (5, :closurelt))
      if isa(inargs[i], Function)
        setfield!(
          parameter.diffmethods,
          field,
          nkeys == 0 ? inargs[i] : _x::Real -> inargs[i](_x, Any[s.value for s in parameter.states])
        )
      end
    end

    for (i, returnname, diffresult, diffmethod) in (
      (6, :gradloglikelihood, :resultll, :closurell),
      (7, :gradlogprior, :resultlp, :closurelp),
      (8, :gradlogtarget, :resultlt, :closurelt)
    )
      if !isa(inargs[i], Function) && isa(inargs[i-3], Function)
        outargs[i] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
          setfield!(
            _state,
            returnname,
            forward_autodiff_derivative(
              getfield(_state.diffstate, diffresult), getfield(_state.diffmethods, diffmethod), _state.value
            )
          )
      end
    end

    if !isa(inargs[15], Function) && isa(inargs[5], Function)
      outargs[15] = (_state::BasicContUnvParameterState, _states::VariableStateVector) ->
        (_state.logtarget, _state.gradlogtarget) =
          forward_autodiff_upto_derivative(_state.diffstate.resultlt, _state.diffmethods.closurelt, _state.value)
    end
  end

  BasicContUnvParameter!(parameter, outargs...)

  parameter
end

value_support(::Type{BasicContUnvParameter}) = Continuous
value_support(::BasicContUnvParameter) = Continuous

variate_form(::Type{BasicContUnvParameter}) = Univariate
variate_form(::BasicContUnvParameter) = Univariate

default_state_type(::BasicContUnvParameter) = BasicContUnvParameterState

default_state(variable::BasicContUnvParameter, value::N, outopts::Dict) where {N<:Real} =
  BasicContUnvParameterState(
    value,
    (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[],
    variable.diffmethods,
    variable.diffopts
  )

show(io::IO, ::Type{BasicContUnvParameter}) = print(io, "BasicContUnvParameter")
