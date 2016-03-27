type BasicDiscUnvParameter{S<:VariableState} <: Parameter{Discrete, Univariate}
  key::Symbol
  index::Int
  pdf::Union{DiscreteUnivariateDistribution, Void}
  prior::Union{DiscreteUnivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  states::Vector{S}

  function BasicDiscUnvParameter(
    key::Symbol,
    index::Int,
    pdf::Union{DiscreteUnivariateDistribution, Void},
    prior::Union{DiscreteUnivariateDistribution, Void},
    setpdf::Union{Function, Void},
    setprior::Union{Function, Void},
    ll::Union{Function, Void},
    lp::Union{Function, Void},
    lt::Union{Function, Void},
    states::Vector{S}
  )
    args = (setpdf, setprior, ll, lp, lt)
    fnames = fieldnames(BasicDiscUnvParameter)[5:9]

    # Check that all generic functions have correct signature
    for i in 1:5
      if isa(args[i], Function) &&
        isgeneric(args[i]) &&
        !(any([method_exists(args[i], (BasicDiscUnvParameterState, Vector{S})) for S in subtypes(VariableState)]))
        error("$(fnames[i]) has wrong signature")
      end
    end

    new(key, index, pdf, prior, setpdf, setprior, ll, lp, lt, states)
  end
end

BasicDiscUnvParameter{S<:VariableState}(
  key::Symbol,
  index::Int,
  pdf::Union{DiscreteUnivariateDistribution, Void},
  prior::Union{DiscreteUnivariateDistribution, Void},
  setpdf::Union{Function, Void},
  setprior::Union{Function, Void},
  ll::Union{Function, Void},
  lp::Union{Function, Void},
  lt::Union{Function, Void},
  states::Vector{S}
) =
  BasicDiscUnvParameter{S}(key, index, pdf, prior, setpdf, setprior, ll, lp, lt, states)

function BasicDiscUnvParameter!(
  parameter::BasicDiscUnvParameter,
  setpdf::Union{Function, Void},
  setprior::Union{Function, Void},
  ll::Union{Function, Void},
  lp::Union{Function, Void},
  lt::Union{Function, Void}
)
  args = (setpdf, setprior, ll, lp, lt)

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

  # Define loglikelihood!
  setfield!(
    parameter,
    :loglikelihood!,
    if isa(args[3], Function)
      eval(codegen_method(parameter, args[3]))
    else
      nothing
    end
  )

  # Define logprior!
  setfield!(
    parameter,
    :logprior!,
    if isa(args[4], Function)
      eval(codegen_method(parameter, args[4]))
    else
      if (
          isa(parameter.prior, DiscreteUnivariateDistribution) &&
          method_exists(logpdf, (typeof(parameter.prior), eltype(parameter.prior)))
        ) ||
        isa(args[2], Function)
        eval(codegen_method_via_distribution(parameter, :prior, logpdf, :logprior))
      else
        nothing
      end
    end
  )

  # Define logtarget!
  setfield!(
    parameter,
    :logtarget!,
    if isa(args[5], Function)
      eval(codegen_method(parameter, args[5]))
    else
      if isa(args[3], Function) && isa(getfield(parameter, :logprior!), Function)
        eval(codegen_method_via_sum(parameter, :loglikelihood!, :logprior!, :logtarget, :loglikelihood, :logprior))
      elseif (
          isa(parameter.pdf, DiscreteUnivariateDistribution) &&
          method_exists(logpdf, (typeof(parameter.pdf), eltype(parameter.pdf)))
        ) ||
        isa(args[1], Function)
        eval(codegen_method_via_distribution(parameter, :pdf, logpdf, :logtarget))
      else
        nothing
      end
    end
  )
end

function BasicDiscUnvParameter{S<:VariableState}(
  key::Symbol,
  index::Int;
  pdf::Union{DiscreteUnivariateDistribution, Void}=nothing,
  prior::Union{DiscreteUnivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  states::Vector{S}=VariableState[]
)
  parameter = BasicDiscUnvParameter(key, index, pdf, prior, fill(nothing, 5)..., states)
  BasicDiscUnvParameter!(parameter, setpdf, setprior, loglikelihood, logprior, logtarget)
  parameter
end

function BasicDiscUnvParameter{S<:VariableState}(
  key::Symbol;
  index::Int=0,
  pdf::Union{DiscreteUnivariateDistribution, Void}=nothing,
  prior::Union{DiscreteUnivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  states::Vector{S}=VariableState[],
  nkeys::Int=0,
  vfarg::Bool=false
)
  inargs = (setpdf, setprior, loglikelihood, logprior, logtarget)

  fnames = Array(Any, 5)
  fnames[1:2] = fill(Symbol[], 2)
  fnames[3:5] = [Symbol[f] for f in fieldnames(BasicDiscUnvParameterState)[2:4]]

  @assert nkeys >= 0 "nkeys must be non-negative, got $nkeys"
  
  parameter = BasicDiscUnvParameter(key, index, pdf, prior, fill(nothing, 5)..., states)

  outargs = Union{Function, Void}[nothing for i in 1:5]

  for i in 1:5
    if isa(inargs[i], Function)
      outargs[i] = eval(codegen_internal_variable_method(inargs[i], fnames[i], nkeys, vfarg))
    end
  end

  BasicDiscUnvParameter!(parameter, outargs...)

  parameter
end

value_support{S<:VariableState}(::Type{BasicDiscUnvParameter{S}}) = Discrete
value_support(::BasicDiscUnvParameter) = Discrete

variate_form{S<:VariableState}(::Type{BasicDiscUnvParameter{S}}) = Univariate
variate_form(::BasicDiscUnvParameter) = Univariate

default_state_type(::BasicDiscUnvParameter) = BasicDiscUnvParameterState

default_state{N<:Integer}(variable::BasicDiscUnvParameter, value::N, outopts::Dict) =
  BasicDiscUnvParameterState(
    value, (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[]
  )

Base.show{S<:VariableState}(io::IO, ::Type{BasicDiscUnvParameter{S}}) = print(io, "BasicDiscUnvParameter")
Base.writemime{S<:VariableState}(io::IO, ::MIME"text/plain", t::Type{BasicDiscUnvParameter{S}}) = show(io, t)
