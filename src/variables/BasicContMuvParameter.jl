### BasicContMuvParameter

type BasicContMuvParameter <: Parameter{Continuous, Multivariate}
  key::Symbol
  index::Int
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

  function BasicContMuvParameter(
    key::Symbol,
    index::Int,
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
    uptodtlt::Union{Function, Void}
  )
    instance = new()
    instance.key = key
    instance.index = index
    instance.pdf = pdf
    instance.prior = prior

    args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
    fnames = fieldnames(BasicContMuvParameter)[5:21]

    # Check that all generic functions have correct signature
    for i = 1:17
      if isa(args[i], Function) &&
        isgeneric(args[i]) &&
        !any(
          [method_exists(args[i], (BasicContMuvParameterState, Vector{S})) for S in subtypes(VariableState)]
        )
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(args[i], Function)
          eval(codegen_setdistribution_continuous_multivariate_parameter(instance, distribution, args[i]))
        else
          args[i]
        end
      )
    end

    # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
    instance.loglikelihood! = args[3]
    instance.gradloglikelihood! = args[6]

    # Define logprior! (i = 4) and gradlogprior! (i = 7)
    # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
    for (i , ppfield, spfield, f) in (
      (4, :logprior!, :logprior, logpdf), (7, :gradlogprior!, :gradlogprior, gradlogpdf)
    )
      setfield!(
        instance,
        ppfield,
        if args[i] == nothing && (
          (
            isa(prior, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(prior), Vector{eltype(prior)}))
          ) ||
          isa(args[2], Function)
        )
          eval(codegen_setfield_via_distribution_continuous_multivariate_parameter(instance, spfield, :prior, f))
        else
          args[i]
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
        if args[i] == nothing
          if isa(args[i-2], Function) && isa(getfield(instance, ppfield), Function)
            eval(codegen_setfield_via_sum_continuous_multivariate_parameter(
              instance, plfield, ppfield, stfield, slfield, spfield
            ))
          elseif (
              isa(pdf, ContinuousMultivariateDistribution) && method_exists(f, (typeof(pdf), Vector{eltype(pdf)}))
            ) ||
            isa(args[1], Function)
            eval(codegen_setfield_via_distribution_continuous_multivariate_parameter(instance, stfield, :pdf, f))
          end
        else
          args[i]
        end
      )
    end

    # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
    instance.tensorloglikelihood! = args[9]
    instance.dtensorloglikelihood! = args[12]

    # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
    instance.tensorlogprior! = args[10]
    instance.dtensorlogprior! = args[13]

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
        if args[i] == nothing && isa(args[i-2], Function) && isa(args[i-1], Function)
          eval(codegen_setfield_via_sum_continuous_multivariate_parameter(
            instance, plfield, ppfield, stfield, slfield, spfield
          ))
        else
          args[i]
        end
      )
    end

    # Define uptogradlogtarget!
    setfield!(
      instance,
      :uptogradlogtarget!,
      if args[15] == nothing && isa(instance.logtarget!, Function) && isa(instance.gradlogtarget!, Function)
        eval(codegen_setuptofields_continuous_multivariate_parameter(instance, [:logtarget!, :gradlogtarget!]))
      else
        args[15]
      end
    )

    # Define uptotensorlogtarget!
    setfield!(
      instance,
      :uptotensorlogtarget!,
      if args[16] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function)
        eval(codegen_setuptofields_continuous_multivariate_parameter(
          instance, [:logtarget!, :gradlogtarget!, :tensorlogtarget!]
        ))
      else
        args[16]
      end
    )

    # Define uptodtensorlogtarget!
    setfield!(
      instance,
      :uptodtensorlogtarget!,
      if args[17] == nothing &&
        isa(instance.logtarget!, Function) &&
        isa(instance.gradlogtarget!, Function) &&
        isa(instance.tensorlogtarget!, Function) &&
        isa(instance.dtensorlogtarget!, Function)
        eval(codegen_setuptofields_continuous_multivariate_parameter(
          instance, [:logtarget!, :gradlogtarget!, :tensorlogtarget!, :dtensorlogtarget!]
        ))
      else
        args[17]
      end
    )

    instance
  end
end

BasicContMuvParameter(
  key::Symbol,
  index::Int=0;
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
  uptodtensorlogtarget::Union{Function, Void}=nothing
) =
  BasicContMuvParameter(
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
    uptodtensorlogtarget
  )

function BasicContMuvParameter(
  key::Vector{Symbol},
  index::Int;
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
  uptodtensorlogtarget::Union{Function, Void}=nothing
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
  fnames[3:14] = [Symbol[f] for f in fieldnames(BasicContMuvParameterState)[2:13]]
  for i in 1:3
    fnames[14+i] = fnames[5:3:(5+i*3)]
  end

  for i in 1:17
    if inargs[i] == nothing
      outargs[i] = nothing
    elseif isa(inargs[i], Function)
      if isgeneric(inargs[i])
        if any([method_exists(inargs[i], (T,)) for T in
          [Any; [Vector{N} for N in (Number, Real, AbstractFloat, BigFloat, Float64, Float32, Float16)]]
        ])
          outargs[i] = eval(codegen_internal_variable_method(inargs[i], fnames[i]))
        elseif any([method_exists(inargs[i], (T, Dict)) for T in
          [Any; [Vector{N} for N in (Number, Real, AbstractFloat, BigFloat, Float64, Float32, Float16)]]
        ])
          outargs[i] = eval(codegen_internal_variable_method(inargs[i], fnames[i], key, index))
        else
          error("Function $(f[i]) has wrong signature")
        end
      else
        error("BasicContMuvParameter with vector key input argument works only with generic input methods")
      end
    end
  end

  BasicContMuvParameter(key[index], index, pdf, prior, outargs...)
end

function codegen_setdistribution_continuous_multivariate_parameter(
  parameter::BasicContMuvParameter,
  distribution::Symbol,
  f::Function
)
  body = :(setfield!($(parameter), $(QuoteNode(distribution)), $(f)(_state, _states)))
  @gensym setdistribution_continuous_multivariate_parameter
  quote
    function $setdistribution_continuous_multivariate_parameter{S<:VariableState}(
      _state::BasicContMuvParameterState,
      _states::Vector{S})
      $(body)
    end
  end
end

function codegen_setfield_via_distribution_continuous_multivariate_parameter(
  parameter::BasicContMuvParameter,
  field::Symbol,
  distribution::Symbol,
  f::Function
)
  body = :(setfield!(_state, $(QuoteNode(field)), $(f)(getfield($(parameter), $(QuoteNode(distribution))), _state.value)))
  @gensym codegen_setfield_via_distribution_continuous_multivariate_parameter
  quote
    function $codegen_setfield_via_distribution_continuous_multivariate_parameter{S<:VariableState}(
      _state::BasicContMuvParameterState,
      _states::Vector{S})
      $(body)
    end
  end
end

function codegen_setfield_via_sum_continuous_multivariate_parameter(
  parameter::BasicContMuvParameter,
  plfield::Symbol,
  ppfield::Symbol,
  stfield::Symbol,
  slfield::Symbol,
  spfield::Symbol
)
  body = []

  push!(body, :(getfield($(parameter), $(QuoteNode(plfield)))(_state, _states)))
  push!(body, :(getfield($(parameter), $(QuoteNode(ppfield)))(_state, _states)))
  push!(body, :(setfield!(
    _state,
    $(QuoteNode(stfield)),
    getfield(_state, $(QuoteNode(slfield)))+getfield(_state, $(QuoteNode(spfield)))))
  )

  @gensym codegen_setfield_via_sum_continuous_multivariate_parameter

  quote
    function $codegen_setfield_via_sum_continuous_multivariate_parameter{S<:VariableState}(
      _state::BasicContMuvParameterState,
      _states::Vector{S})
      $(body...)
    end
  end
end

function codegen_setuptofields_continuous_multivariate_parameter(
  parameter::BasicContMuvParameter,
  fields::Vector{Symbol}
)
  body = []
  local f::Symbol

  for i in 1:length(fields)
    f = fields[i]
    push!(body, :(getfield($(parameter), $(QuoteNode(f)))(_state, _states)))
  end

  @gensym codegen_setuptofields_continuous_multivariate_parameter

  quote
    function $codegen_setuptofields_continuous_multivariate_parameter{S<:VariableState}(
      _state::BasicContMuvParameterState,
      _states::Vector{S})
      $(body...)
    end
  end
end

value_support(s::Type{BasicContMuvParameter}) = Continuous
value_support(s::BasicContMuvParameter) = Continuous

variate_form(s::Type{BasicContMuvParameter}) = Multivariate
variate_form(s::BasicContMuvParameter) = Multivariate

function default_state{N<:Real}(variable::BasicContMuvParameter, value::Vector{N}, outopts::Dict)
  augment!(outopts)
  BasicContMuvParameterState(
    value,
    [getfield(variable, fieldnames(BasicContMuvParameter)[i]) == nothing ? false : true for i in 10:18],
    in(:accept, outopts[:diagnostics]) ? [:accept] : Symbol[]
  )
end
