### BasicContUnvParameter

# Guidelines for usage of inner constructors of continuous parameter types:
# 1) Function fields have higher priority than implicitly derived definitions via the pdf field
# 2) Target-related fields have higher priority than implicitly derived likelihood+prior fields
# 3) Upto-related fields have higher priority than implicitly derived Function tuples

### BasicContUnvParameter

type BasicContUnvParameter{S<:VariableState} <: Parameter{Continuous, Univariate}
  key::Symbol
  index::Int
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
  states::Vector{S}

  function BasicContUnvParameter(
    key::Symbol,
    index::Int,
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
    states::Vector{S}
  )
    instance = new()
    instance.key = key
    instance.index = index
    instance.pdf = pdf
    instance.prior = prior
    instance.states = states

    args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
    fnames = fieldnames(BasicContUnvParameter)[5:21]

    # Check that all generic functions have correct signature
    for i = 1:17
      if isa(args[i], Function) &&
        isgeneric(args[i]) &&
        !(any([method_exists(args[i], (BasicContUnvParameterState, Vector{S})) for S in subtypes(VariableState)]))
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(args[i], Function)
          eval(codegen_setfield_basiccontunvparameter(instance, distribution, args[i]))
        else
          nothing
        end
      )
    end

    # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
    instance.loglikelihood! = if isa(args[3], Function)
      eval(codegen_method_basiccontunvparameter(instance, args[3]))
    else
      nothing
    end
    instance.gradloglikelihood! = if isa(args[6], Function)
      eval(codegen_method_basiccontunvparameter(instance, args[6]))
    else
      nothing
    end

    # Define logprior! (i = 4) and gradlogprior! (i = 7)
    # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
    for (i , ppfield, spfield, f) in (
      (4, :logprior!, :logprior, logpdf), (7, :gradlogprior!, :gradlogprior, gradlogpdf)
    )
      setfield!(
        instance,
        ppfield,
        if isa(args[i], Function)
          eval(codegen_method_basiccontunvparameter(instance, args[i]))
        else
          if (isa(prior, ContinuousUnivariateDistribution) && method_exists(f, (typeof(prior), eltype(prior)))) ||
            isa(args[2], Function)
            eval(codegen_method_via_distribution_basiccontunvparameter(instance, :prior, f, spfield))
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
      (
        8,
        :gradlogtarget!, :gradloglikelihood!, :gradlogprior!,
        :gradlogtarget, :gradloglikelihood, :gradlogprior,
        gradlogpdf
      )
    )
      setfield!(
        instance,
        ptfield,
        if isa(args[i], Function)
          eval(codegen_method_basiccontunvparameter(instance, args[i]))
        else
          if isa(args[i-2], Function) && isa(getfield(instance, ppfield), Function)
            eval(codegen_method_via_sum_basiccontunvparameter(instance, plfield, ppfield, stfield, slfield, spfield))
          elseif (isa(pdf, ContinuousUnivariateDistribution) && method_exists(f, (typeof(pdf), eltype(pdf)))) ||
            isa(args[1], Function)
            eval(codegen_method_via_distribution_basiccontunvparameter(instance, :pdf, f, stfield))
          else
            nothing
          end
        end
      )
    end

    # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
    instance.tensorloglikelihood! = if isa(args[9], Function)
      eval(codegen_method_basiccontunvparameter(instance, args[9]))
    else
      nothing
    end
    instance.dtensorloglikelihood! = if isa(args[12], Function)
      eval(codegen_method_basiccontunvparameter(instance, args[12]))
    else
      nothing
    end

    # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
    instance.tensorlogprior! = if isa(args[10], Function)
      eval(codegen_method_basiccontunvparameter(instance, args[10]))
    else
      nothing
    end
    instance.dtensorlogprior! = if isa(args[13], Function)
      eval(codegen_method_basiccontunvparameter(instance, args[13]))
    else
      nothing
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
        instance,
        ptfield,
        if isa(args[i], Function)
          eval(codegen_method_basiccontunvparameter(instance, args[i]))
        else
          if isa(args[i-2], Function) && isa(args[i-1], Function)
            eval(codegen_method_via_sum_basiccontunvparameter(instance, plfield, ppfield, stfield, slfield, spfield))
          else
            nothing
          end
        end
      )
    end

    # Define uptogradlogtarget!
    setfield!(
      instance,
      :uptogradlogtarget!,
      if isa(args[15], Function)
        eval(codegen_method_basiccontunvparameter(instance, args[15]))
      else
        if isa(instance.logtarget!, Function) && isa(instance.gradlogtarget!, Function)
          eval(codegen_uptomethods_basiccontunvparameter(instance, [:logtarget!, :gradlogtarget!]))
        else
          nothing
        end
      end
    )

    # Define uptotensorlogtarget!
    setfield!(
      instance,
      :uptotensorlogtarget!,
      if isa(args[16], Function)
        eval(codegen_method_basiccontunvparameter(instance, args[16]))
      else
        if isa(instance.logtarget!, Function) &&
          isa(instance.gradlogtarget!, Function) &&
          isa(instance.tensorlogtarget!, Function)
          eval(codegen_uptomethods_basiccontunvparameter(instance, [:logtarget!, :gradlogtarget!, :tensorlogtarget!]))
        else
          nothing
        end
      end
    )

    # Define uptodtensorlogtarget!
    setfield!(
      instance,
      :uptodtensorlogtarget!,
      if isa(args[17], Function)
        eval(codegen_method_basiccontunvparameter(instance, args[17]))
      else
        if isa(instance.logtarget!, Function) &&
          isa(instance.gradlogtarget!, Function) &&
          isa(instance.tensorlogtarget!, Function) &&
          isa(instance.dtensorlogtarget!, Function)
          eval(codegen_uptomethods_basiccontunvparameter(
            instance, [:logtarget!, :gradlogtarget!, :tensorlogtarget!, :dtensorlogtarget!]
          ))
        else
          nothing
        end
      end
    )

    instance
  end
end

BasicContUnvParameter{S<:VariableState}(
  key::Symbol,
  index::Int,
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
  states::Vector{S}
) =
  BasicContUnvParameter{S}(
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
    states
  )

BasicContUnvParameter{S<:VariableState}(
  key::Symbol,
  index::Int=0;
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
  states::Vector{S}=VariableState[]
) =
  BasicContUnvParameter(
    key,
    index,
    pdf,
    prior,
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
    uptodtensorlogtarget,
    states
  )

function BasicContUnvParameter(
  key::Vector{Symbol},
  index::Int;
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
  nkeys::Int=length(key),
  vfarg::Bool=false
)
  outargs = Array(Union{Function, Void}, 17)

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

  fnames = Array(Any, 17)
  fnames[1:2] = fill(Symbol[], 2)
  fnames[3:14] = [Symbol[f] for f in fieldnames(BasicContUnvParameterState)[2:13]]
  for i in 1:3
    fnames[14+i] = fnames[5:3:(5+i*3)]
  end

  for i in 1:17
    if inargs[i] == nothing
      outargs[i] = nothing
    elseif isa(inargs[i], Function)
      outargs[i] = eval(codegen_internal_variable_method(inargs[i], fnames[i], nkeys, vfarg))
    end
  end

  BasicContUnvParameter(key[index], index, pdf, prior, outargs..., VariableState[])
end

function codegen_setfield_basiccontunvparameter(parameter::BasicContUnvParameter, field::Symbol, f::Function)
  @gensym codegen_setfield_basiccontunvparameter
  quote
    function $codegen_setfield_basiccontunvparameter(_state::BasicContUnvParameterState)
      setfield!($(parameter), $(QuoteNode(field)), $(f)(_state, $(parameter).states))
    end
  end
end

function codegen_method_basiccontunvparameter(parameter::BasicContUnvParameter, f::Function)
  @gensym codegen_method_basiccontunvparameter
  quote
    function $codegen_method_basiccontunvparameter(_state::BasicContUnvParameterState)
      $(f)(_state, $(parameter).states)
    end
  end
end

function codegen_method_via_distribution_basiccontunvparameter(
  parameter::BasicContUnvParameter,
  distribution::Symbol,
  f::Function,
  field::Symbol
)
  @gensym codegen_method_via_distribution_basiccontunvparameter
  quote
    function $codegen_method_via_distribution_basiccontunvparameter(_state::BasicContUnvParameterState)
      setfield!(_state, $(QuoteNode(field)), $(f)(getfield($(parameter), $(QuoteNode(distribution))), _state.value))
    end
  end
end

function codegen_method_via_sum_basiccontunvparameter(
  parameter::BasicContUnvParameter,
  plfield::Symbol,
  ppfield::Symbol,
  stfield::Symbol,
  slfield::Symbol,
  spfield::Symbol
)
  body = []

  push!(body, :(getfield($(parameter), $(QuoteNode(plfield)))(_state)))
  push!(body, :(getfield($(parameter), $(QuoteNode(ppfield)))(_state)))
  push!(body, :(setfield!(
    _state,
    $(QuoteNode(stfield)),
    getfield(_state, $(QuoteNode(slfield)))+getfield(_state, $(QuoteNode(spfield)))))
  )

  @gensym codegen_method_via_sum_basiccontunvparameter

  quote
    function $codegen_method_via_sum_basiccontunvparameter(_state::BasicContUnvParameterState)
      $(body...)
    end
  end
end

function codegen_uptomethods_basiccontunvparameter(parameter::BasicContUnvParameter, fields::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:length(fields)
    f = fields[i]
    push!(body, :(getfield($(parameter), $(QuoteNode(f)))(_state)))
  end

  @gensym codegen_uptomethods_basiccontunvparameter

  quote
    function $codegen_uptomethods_basiccontunvparameter(_state::BasicContUnvParameterState)
      $(body...)
    end
  end
end

value_support(s::Type{BasicContUnvParameter}) = Continuous
value_support(s::BasicContUnvParameter) = Continuous

variate_form(s::Type{BasicContUnvParameter}) = Univariate
variate_form(s::BasicContUnvParameter) = Univariate

default_state{N<:Real}(variable::BasicContUnvParameter, value::N, outopts::Dict) =
  BasicContUnvParameterState(
    value,
    (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[]
  )
