### BasicDiscMuvParameter

type BasicDiscMuvParameter <: Parameter{Discrete, Multivariate}
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
        isgeneric(args[i]) &&
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
          eval(codegen_setfield(instance, distribution, args[i]))
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
        eval(codegen_closure(instance, args[3]))
      else
        nothing
      end
    )

    # Define logprior!
    setfield!(
      instance,
      :logprior!,
      if isa(args[4], Function)
        eval(codegen_closure(instance, args[4]))
      else
        if (isa(prior, DiscreteMultivariateDistribution) && method_exists(logpdf, (typeof(prior), Vector{eltype(prior)}))) ||
          isa(args[2], Function)
          eval(codegen_target_closure_via_distribution(instance, :prior, logpdf, :logprior))
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
        eval(codegen_closure(instance, args[5]))
      else
        if isa(args[3], Function) && isa(getfield(instance, :logprior!), Function)
          eval(codegen_sumtarget_closure(instance, :loglikelihood!, :logprior!, :logtarget, :loglikelihood, :logprior))
        elseif (isa(pdf, DiscreteMultivariateDistribution) && method_exists(logpdf, (typeof(pdf), Vector{eltype(pdf)}))) ||
          isa(args[1], Function)
          eval(codegen_target_closure_via_distribution(instance, :pdf, logpdf, :logtarget))
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

  fnames = Array(Any, 5)
  fnames[1:2] = fill(Symbol[], 2)
  fnames[3:5] = [Symbol[f] for f in fieldnames(BasicDiscMuvParameterState)[2:4]]

  @assert nkeys >= 0 "nkeys must be non-negative, got $nkeys"

  outargs = Union{Function, Void}[nothing for i in 1:5]

  for i in 1:5
    if isa(inargs[i], Function)
      outargs[i] = eval(
        codegen_lowlevel_variable_method(inargs[i], :BasicDiscMuvParameterState, true, fnames[i], nkeys, vfarg)
      )
    end
  end

  BasicDiscMuvParameter(key, index, pdf, prior, outargs..., states)
end

value_support(::Type{BasicDiscMuvParameter}) = Discrete
value_support(::BasicDiscMuvParameter) = Discrete

variate_form(::Type{BasicDiscMuvParameter}) = Multivariate
variate_form(::BasicDiscMuvParameter) = Multivariate

default_state_type(::BasicDiscMuvParameter) = BasicDiscMuvParameterState

default_state{N<:Integer}(variable::BasicDiscMuvParameter, value::Vector{N}, outopts::Dict) =
  BasicDiscMuvParameterState(
    value, (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[]
  )

Base.show(io::IO, ::Type{BasicDiscMuvParameter}) = print(io, "BasicDiscMuvParameter")
Base.writemime(io::IO, ::MIME"text/plain", t::Type{BasicDiscMuvParameter}) = show(io, t)
