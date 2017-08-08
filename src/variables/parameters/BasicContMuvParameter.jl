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
        eval(codegen_setfield(parameter, distribution, args[i]))
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
        eval(codegen_closure(parameter, args[i]))
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
        eval(codegen_closure(parameter, args[i]))
      else
        if (
            isa(parameter.prior, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(parameter.prior), Vector{eltype(parameter.prior)}))
          ) ||
          isa(args[2], Function)
          eval(codegen_target_closure_via_distribution(parameter, :prior, f, spfield))
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
        eval(codegen_closure(parameter, args[i]))
      else
        if isa(args[i-2], Function) && isa(getfield(parameter, ppfield), Function)
          eval(codegen_sumtarget_closure(parameter, plfield, ppfield, stfield, slfield, spfield))
        elseif (
            isa(parameter.pdf, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(parameter.pdf), Vector{eltype(parameter.pdf)}))
          ) ||
          isa(args[1], Function)
          eval(codegen_target_closure_via_distribution(parameter, :pdf, f, stfield))
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
        eval(codegen_closure(parameter, args[i]))
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
        eval(codegen_closure(parameter, args[i]))
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
        eval(codegen_closure(parameter, args[i]))
      else
        if isa(args[i-2], Function) && isa(args[i-1], Function)
          eval(codegen_sumtarget_closure(parameter, plfield, ppfield, stfield, slfield, spfield))
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
      eval(codegen_closure(parameter, args[15]))
    else
      if isa(parameter.logtarget!, Function) && isa(parameter.gradlogtarget!, Function)
        eval(codegen_uptotarget_closures(parameter, [:logtarget!, :gradlogtarget!]))
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
      eval(codegen_closure(parameter, args[16]))
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function)
        eval(codegen_uptotarget_closures(parameter, [:logtarget!, :gradlogtarget!, :tensorlogtarget!]))
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
      eval(codegen_closure(parameter, args[17]))
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function) &&
        isa(parameter.dtensorlogtarget!, Function)
        eval(codegen_uptotarget_closures(parameter, [:logtarget!, :gradlogtarget!, :tensorlogtarget!, :dtensorlogtarget!]))
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

  fnames = Array{Any}(17)
  fnames[1:2] = fill(Symbol[], 2)
  fnames[3:14] = [Symbol[f] for f in fieldnames(BasicContMuvParameterState)[2:13]]
  for i in 1:3
    fnames[14+i] = Symbol[fnames[j][1] for j in 5:3:(5+i*3)]
  end

  if nkeys > 0
    if diffopts != nothing && vfarg
      error("In the case of autodiff, if nkeys is not 0, then vfarg must be false")
    end
  elseif nkeys < 0
    "nkeys must be non-negative, got $nkeys"
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

  for i in 1:17
    if isa(inargs[i], Function)
      outargs[i] = eval(
        codegen_lowlevel_variable_method(
          inargs[i], statetype=:BasicContMuvParameterState, returns=fnames[i], vfarg=vfarg, nkeys=nkeys
        )
      )
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
      outargs[i] = eval(codegen_setfield(parameter, field, distribution, f))
    end
  end

  if diffopts != nothing
    for (i, field) in ((3, :closurell), (4, :closurelp), (5, :closurelt))
      if isa(inargs[i], Function)
        setfield!(
          parameter.diffmethods,
          field,
          nkeys == 0 ? inargs[i] : eval(codegen_internal_autodiff_closure(parameter, inargs[i], nkeys))
        )
      end
    end

    diffmethods = diffopts.mode == :reverse ? (:tapegll, :tapeglp, :tapeglt) : (:closurell, :closurelp, :closurelt)

    for (i, diffresult, diffmethod, diffconfig) in (
      (6, :resultll, diffmethods[1], :cfggll),
      (7, :resultlp, diffmethods[2], :cfgglp),
      (8, :resultlt, diffmethods[3], :cfgglt)
    )
      if !isa(inargs[i], Function) && isa(inargs[i-3], Function)
        outargs[i] = eval(codegen_lowlevel_variable_method(
          eval(codegen_autodiff_function(diffopts.mode, :gradient)),
          statetype=:BasicContMuvParameterState,
          returns=fnames[i],
          diffresult=diffresult,
          diffmethod=diffmethod,
          diffconfig=(diffopts.mode == :reverse ? nothing : diffconfig)
        ))
      end
    end

    if !isa(inargs[15], Function) && isa(inargs[5], Function)
      outargs[15] = eval(codegen_lowlevel_variable_method(
        eval(codegen_autodiff_uptofunction(diffopts.mode, :gradient)),
        statetype=:BasicContMuvParameterState,
        returns=fnames[15],
        diffresult=:resultlt,
        diffmethod=diffmethods[3],
        diffconfig=(diffopts.mode == :reverse ? nothing : :cfgglt)
      ))
    end

    if diffopts.order == 2
      diffmethods = diffopts.mode == :reverse ? (:tapetll, :tapetlp, :tapetlt) : (:closurell, :closurelp, :closurelt)

      for (i, diffresult, diffmethod, diffconfig) in (
        (9, :resultll, diffmethods[1], :cfgtll),
        (10, :resultlp, diffmethods[2], :cfgtlp),
        (11, :resultlt, diffmethods[3], :cfgtlt)
      )
        if !isa(inargs[i], Function) && isa(inargs[i-6], Function)
          outargs[i] = eval(codegen_lowlevel_variable_method(
            eval(codegen_autodiff_target(diffopts.mode, :hessian)),
            statetype=:BasicContMuvParameterState,
            returns=fnames[i],
            diffresult=diffresult,
            diffmethod=diffmethod,
            diffconfig=(diffopts.mode == :reverse ? nothing : diffconfig)
          ))
        end
      end

      if !isa(inargs[16], Function) && isa(inargs[5], Function)
        outargs[16] = eval(codegen_lowlevel_variable_method(
          eval(codegen_autodiff_uptotarget(diffopts.mode, :hessian)),
          statetype=:BasicContMuvParameterState,
          returns=fnames[16],
          diffresult=:resultlt,
          diffmethod=diffmethods[3],
          diffconfig=(diffopts.mode == :reverse ? nothing : :cfgtlt)
        ))
      end
    end
  end

  BasicContMuvParameter!(parameter, outargs...)

  parameter
end

function codegen_internal_autodiff_closure(parameter::BasicContMuvParameter, f::Function, nkeys::Integer)
  fstatesarg = [Expr(:ref, :Any, [:($(parameter).states[$i].value) for i in 1:nkeys]...)]

  arg = parameter.diffopts.mode == :reverse ? :_x : :(_x::Vector)

  @gensym internal_autodiff_closure

  quote
    function $internal_autodiff_closure($arg)
      $(f)(_x, $(fstatesarg...))
    end
  end
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
