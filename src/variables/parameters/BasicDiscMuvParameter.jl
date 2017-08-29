### BasicDiscMuvParameter

mutable struct BasicDiscMuvParameter <: Parameter{Discrete, Multivariate}
  key::Symbol
  index::Integer
  pdf::Union{DiscreteMultivariateDistribution, Void}
  prior::Union{DiscreteMultivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  states::VariableStateVector

  function BasicDiscMuvParameter(
    key::Symbol,
    index::Integer,
    pdf::Union{DiscreteMultivariateDistribution, Void},
    prior::Union{DiscreteMultivariateDistribution, Void},
    setpdf::Union{Function, Void},
    setprior::Union{Function, Void},
    ll::Union{Function, Void},
    lp::Union{Function, Void},
    lt::Union{Function, Void},
    states::VariableStateVector
  )
    instance = new()

    instance.key = key
    instance.index = index
    instance.pdf = pdf
    instance.prior = prior

    args = (setpdf, setprior, ll, lp, lt)
    fnames = fieldnames(BasicDiscMuvParameter)[5:9]

    # Check that all generic functions have correct signature
    for i in 1:5
      if isa(args[i], Function) &&
        isa(args[i], Function) &&
        !method_exists(args[i], (BasicDiscMuvParameterState, VariableStateVector))
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(args[i], Function)
          (_state::BasicDiscMuvParameterState, _states::VariableStateVector) ->
            setfield!(instance, distribution, args[i](_state, _states))
        else
          nothing
        end
      )
    end

    # Define loglikelihood!
    setfield!(
      instance,
      :loglikelihood!,
      if isa(args[3], Function)
        _state::BasicDiscMuvParameterState -> args[3](_state, instance.states)
      else
        nothing
      end
    )

    # Define logprior!
    setfield!(
      instance,
      :logprior!,
      if isa(args[4], Function)
        _state::BasicDiscMuvParameterState -> args[4](_state, instance.states)
      else
        if (isa(prior, DiscreteMultivariateDistribution) && method_exists(logpdf, (typeof(prior), Vector{eltype(prior)}))) ||
          isa(args[2], Function)
          _state::BasicDiscMuvParameterState ->
            setfield!(_state, :logprior, logpdf(getfield(instance, :prior), _state.value))
        else
          nothing
        end
      end
    )

    # Define logtarget!
    setfield!(
      instance,
      :logtarget!,
      if isa(args[5], Function)
        _state::BasicDiscMuvParameterState -> args[5](_state, instance.states)
      else
        if isa(args[3], Function) && isa(getfield(instance, :logprior!), Function)
          function (_state::BasicDiscMuvParameterState)
            getfield(instance, :loglikelihood!)(_state)
            getfield(instance, :logprior!)(_state)
            setfield!(_state, :logtarget, getfield(_state, :loglikelihood)+getfield(_state, :logprior))
          end
        elseif (isa(pdf, DiscreteMultivariateDistribution) && method_exists(logpdf, (typeof(pdf), Vector{eltype(pdf)}))) ||
          isa(args[1], Function)
          _state::BasicDiscMuvParameterState -> setfield!(_state, :logtarget, logpdf(getfield(instance, :pdf), _state.value))
        else
          nothing
        end
      end
    )

    instance
  end
end

BasicDiscMuvParameter(key::Symbol, index::Integer=0; signature::Symbol=:high, args...) =
  BasicDiscMuvParameter(key, Val{signature}, index; args...)

BasicDiscMuvParameter(
  key::Symbol,
  ::Type{Val{:low}},
  index::Integer=0;
  pdf::Union{DiscreteMultivariateDistribution, Void}=nothing,
  prior::Union{DiscreteMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  states::VariableStateVector=VariableState[]
) =
  BasicDiscMuvParameter(key, index, pdf, prior, setpdf, setprior, loglikelihood, logprior, logtarget, states)

function BasicDiscMuvParameter(
  key::Symbol,
  ::Type{Val{:high}},
  index::Integer=0;
  pdf::Union{DiscreteMultivariateDistribution, Void}=nothing,
  prior::Union{DiscreteMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Expr, Void}=nothing,
  logprior::Union{Function, Expr, Void}=nothing,
  logtarget::Union{Function, Expr, Void}=nothing,
  states::VariableStateVector=VariableState[],
  nkeys::Integer=0,
  vfarg::Bool=false
)
  inargs = (setpdf, setprior, loglikelihood, logprior, logtarget)

  fnames = fieldnames(BasicDiscMuvParameterState)[2:4]

  if vfarg
    @assert nkeys > 0 "If vfarg=$vfarg, nkeys must be positive, got $nkeys"
  else
    @assert nkeys >= 0 "If vfarg=$vfarg, nkeys must be non-negative, got $nkeys"
  end

  outargs = Union{Function, Void}[nothing for i in 1:5]

  if vfarg
    for i in 1:2
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicDiscMuvParameterState, _states::VariableStateVector) ->
          inargs[i](Any[s.value for s in _states])
      end
    end

    for i in 3:5
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicDiscMuvParameterState, _states::VariableStateVector) ->
          setfield!(_state, fnames[i-2], inargs[i](Any[s.value for s in _states]))
      end
    end
  else
    if nkeys > 0
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscMuvParameterState, _states::VariableStateVector) ->
            inargs[i](_state.value, Any[s.value for s in _states])
        end
      end

      for i in 3:5
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscMuvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value, Any[s.value for s in _states]))
        end
      end
    else
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscMuvParameterState, _states::VariableStateVector) -> inargs[i](_state.value)
        end
      end

      for i in 3:5
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscMuvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value))
        end
      end
    end
  end

  BasicDiscMuvParameter(key, index, pdf, prior, outargs..., states)
end

value_support(::Type{BasicDiscMuvParameter}) = Discrete
value_support(::BasicDiscMuvParameter) = Discrete

variate_form(::Type{BasicDiscMuvParameter}) = Multivariate
variate_form(::BasicDiscMuvParameter) = Multivariate

default_state_type(::BasicDiscMuvParameter) = BasicDiscMuvParameterState

default_state(variable::BasicDiscMuvParameter, value::Vector{N}, outopts::Dict) where {N<:Integer} =
  BasicDiscMuvParameterState(
    value, (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[]
  )

show(io::IO, ::Type{BasicDiscMuvParameter}) = print(io, "BasicDiscMuvParameter")
