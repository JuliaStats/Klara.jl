mutable struct BasicDiscUnvParameter <: Parameter{Discrete, Univariate}
  key::Symbol
  index::Integer
  pdf::Union{DiscreteUnivariateDistribution, Void}
  prior::Union{DiscreteUnivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  states::VariableStateVector

  function BasicDiscUnvParameter(
    key::Symbol,
    index::Integer,
    pdf::Union{DiscreteUnivariateDistribution, Void},
    prior::Union{DiscreteUnivariateDistribution, Void},
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
    fnames = fieldnames(BasicDiscUnvParameter)[5:9]

    # Check that all generic functions have correct signature
    for i in 1:5
      if isa(args[i], Function) &&
        isa(args[i], Function) &&
        !method_exists(args[i], (BasicDiscUnvParameterState, VariableStateVector))
        error("$(fnames[i]) has wrong signature")
      end
    end

    # Define setpdf (i = 1) and setprior (i = 2)
    for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
      setfield!(
        instance,
        setter,
        if isa(args[i], Function)
          (_state::BasicDiscUnvParameterState, _states::VariableStateVector) ->
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
        _state::BasicDiscUnvParameterState -> args[3](_state, instance.states)
      else
        nothing
      end
    )

    # Define logprior!
    setfield!(
      instance,
      :logprior!,
      if isa(args[4], Function)
        _state::BasicDiscUnvParameterState -> args[4](_state, instance.states)
      else
        if (isa(prior, DiscreteUnivariateDistribution) && method_exists(logpdf, (typeof(prior), eltype(prior)))) ||
          isa(args[2], Function)
          _state::BasicDiscUnvParameterState ->
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
        _state::BasicDiscUnvParameterState -> args[5](_state, instance.states)
      else
        if isa(args[3], Function) && isa(getfield(instance, :logprior!), Function)
          function (_state::BasicDiscUnvParameterState)
            getfield(instance, :loglikelihood!)(_state)
            getfield(instance, :logprior!)(_state)
            setfield!(_state, :logtarget, getfield(_state, :loglikelihood)+getfield(_state, :logprior))
          end
        elseif (isa(pdf, DiscreteUnivariateDistribution) && method_exists(logpdf, (typeof(pdf), eltype(pdf)))) ||
          isa(args[1], Function)
          _state::BasicDiscUnvParameterState -> setfield!(_state, :logtarget, logpdf(getfield(instance, :pdf), _state.value))
        else
          nothing
        end
      end
    )

    instance
  end
end

BasicDiscUnvParameter(key::Symbol, index::Integer=0; signature::Symbol=:high, args...) =
  BasicDiscUnvParameter(key, Val{signature}, index; args...)

BasicDiscUnvParameter(
  key::Symbol,
  ::Type{Val{:low}},
  index::Integer=0;
  pdf::Union{DiscreteUnivariateDistribution, Void}=nothing,
  prior::Union{DiscreteUnivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  states::VariableStateVector=VariableState[]
) =
  BasicDiscUnvParameter(key, index, pdf, prior, setpdf, setprior, loglikelihood, logprior, logtarget, states)

function BasicDiscUnvParameter(
  key::Symbol,
  ::Type{Val{:high}},
  index::Integer=0;
  pdf::Union{DiscreteUnivariateDistribution, Void}=nothing,
  prior::Union{DiscreteUnivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  states::VariableStateVector=VariableState[],
  nkeys::Integer=0,
  vfarg::Bool=false
)
  inargs = (setpdf, setprior, loglikelihood, logprior, logtarget)

  fnames = fieldnames(BasicDiscUnvParameterState)[2:4]

  if vfarg
    @assert nkeys > 0 "If vfarg=$vfarg, nkeys must be positive, got $nkeys"
  else
    @assert nkeys >= 0 "If vfarg=$vfarg, nkeys must be non-negative, got $nkeys"
  end

  outargs = Union{Function, Void}[nothing for i in 1:5]

  if vfarg
    for i in 1:2
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicDiscUnvParameterState, _states::VariableStateVector) ->
          inargs[i](Any[s.value for s in _states])
      end
    end

    for i in 3:5
      if isa(inargs[i], Function)
        outargs[i] = (_state::BasicDiscUnvParameterState, _states::VariableStateVector) ->
          setfield!(_state, fnames[i-2], inargs[i](Any[s.value for s in _states]))
      end
    end
  else
    if nkeys > 0
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscUnvParameterState, _states::VariableStateVector) ->
            inargs[i](_state.value, Any[s.value for s in _states])
        end
      end

      for i in 3:5
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscUnvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value, Any[s.value for s in _states]))
        end
      end
    else
      for i in 1:2
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscUnvParameterState, _states::VariableStateVector) -> inargs[i](_state.value)
        end
      end

      for i in 3:5
        if isa(inargs[i], Function)
          outargs[i] = (_state::BasicDiscUnvParameterState, _states::VariableStateVector) ->
            setfield!(_state, fnames[i-2], inargs[i](_state.value))
        end
      end
    end
  end

  BasicDiscUnvParameter(key, index, pdf, prior, outargs..., states)
end

value_support(::Type{BasicDiscUnvParameter}) = Discrete
value_support(::BasicDiscUnvParameter) = Discrete

variate_form(::Type{BasicDiscUnvParameter}) = Univariate
variate_form(::BasicDiscUnvParameter) = Univariate

default_state_type(::BasicDiscUnvParameter) = BasicDiscUnvParameterState

default_state(variable::BasicDiscUnvParameter, value::N, outopts::Dict) where {N<:Integer} =
  BasicDiscUnvParameterState(
    value, (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[]
  )

show(io::IO, ::Type{BasicDiscUnvParameter}) = print(io, "BasicDiscUnvParameter")
