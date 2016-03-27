### BasicDiscMuvParameter

type BasicDiscMuvParameter{S<:VariableState} <: Parameter{Discrete, Multivariate}
  key::Symbol
  index::Int
  pdf::Union{DiscreteMultivariateDistribution, Void}
  prior::Union{DiscreteMultivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  states::Vector{S}

  function BasicDiscMuvParameter(
    key::Symbol,
    index::Int,
    pdf::Union{DiscreteMultivariateDistribution, Void},
    prior::Union{DiscreteMultivariateDistribution, Void},
    setpdf::Union{Function, Void},
    setprior::Union{Function, Void},
    ll::Union{Function, Void},
    lp::Union{Function, Void},
    lt::Union{Function, Void},
    states::Vector{S}
  )
    args = (setpdf, setprior, ll, lp, lt)
    fnames = fieldnames(BasicDiscMuvParameter)[5:9]

    # Check that all generic functions have correct signature
    for i in 1:5
      if isa(args[i], Function) &&
        isgeneric(args[i]) &&
        !(any([method_exists(args[i], (BasicDiscMuvParameterState, Vector{S})) for S in subtypes(VariableState)]))
        error("$(fnames[i]) has wrong signature")
      end
    end

    new(key, index, pdf, prior, setpdf, setprior, ll, lp, lt, states)
  end
end

BasicDiscMuvParameter{S<:VariableState}(
  key::Symbol,
  index::Int,
  pdf::Union{DiscreteMultivariateDistribution, Void},
  prior::Union{DiscreteMultivariateDistribution, Void},
  setpdf::Union{Function, Void},
  setprior::Union{Function, Void},
  ll::Union{Function, Void},
  lp::Union{Function, Void},
  lt::Union{Function, Void},
  states::Vector{S}
) =
  BasicDiscMuvParameter{S}(key, index, pdf, prior, setpdf, setprior, ll, lp, lt, states)

function BasicDiscMuvParameter!(
  parameter::BasicDiscMuvParameter,
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
          isa(parameter.prior, DiscreteMultivariateDistribution) &&
          method_exists(logpdf, (typeof(parameter.prior), Vector{eltype(parameter.prior)}))
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
          isa(parameter.pdf, DiscreteMultivariateDistribution) &&
          method_exists(logpdf, (typeof(parameter.pdf), Vector{eltype(parameter.pdf)}))
        ) ||
        isa(args[1], Function)
        eval(codegen_method_via_distribution(parameter, :pdf, logpdf, :logtarget))
      else
        nothing
      end
    end
  )
end

function BasicDiscMuvParameter{S<:VariableState}(
  key::Symbol,
  index::Int;
  pdf::Union{DiscreteMultivariateDistribution, Void}=nothing,
  prior::Union{DiscreteMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  states::Vector{S}=VariableState[]
)
  parameter = BasicDiscMuvParameter(key, index, pdf, prior, fill(nothing, 5)..., states)
  BasicDiscMuvParameter!(parameter, setpdf, setprior, loglikelihood, logprior, logtarget)
  parameter
end

function BasicDiscMuvParameter{S<:VariableState}(
  key::Symbol;
  index::Int=0,
  pdf::Union{DiscreteMultivariateDistribution, Void}=nothing,
  prior::Union{DiscreteMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Expr, Void}=nothing,
  logprior::Union{Function, Expr, Void}=nothing,
  logtarget::Union{Function, Expr, Void}=nothing,
  states::Vector{S}=VariableState[],
  nkeys::Int=0,
  vfarg::Bool=false
)
  inargs = (setpdf, setprior, loglikelihood, logprior, logtarget)

  fnames = Array(Any, 5)
  fnames[1:2] = fill(Symbol[], 2)
  fnames[3:5] = [Symbol[f] for f in fieldnames(BasicDiscMuvParameterState)[2:4]]

  @assert nkeys >= 0 "nkeys must be non-negative, got $nkeys"
  
  parameter = BasicDiscMuvParameter(key, index, pdf, prior, fill(nothing, 5)..., states)

  outargs = Union{Function, Void}[nothing for i in 1:5]

  for i in 1:5
    if isa(inargs[i], Function)
      outargs[i] = eval(codegen_internal_variable_method(inargs[i], fnames[i], nkeys, vfarg))
    end
  end

  BasicDiscMuvParameter!(parameter, outargs...)

  parameter
end

value_support{S<:VariableState}(::Type{BasicDiscMuvParameter{S}}) = Discrete
value_support(::BasicDiscMuvParameter) = Discrete

variate_form{S<:VariableState}(::Type{BasicDiscMuvParameter{S}}) = Multivariate
variate_form(::BasicDiscMuvParameter) = Multivariate

default_state_type(::BasicDiscMuvParameter) = BasicDiscMuvParameterState

default_state{N<:Integer}(variable::BasicDiscMuvParameter, value::Vector{N}, outopts::Dict) =
  BasicDiscMuvParameterState(
    value, (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[]
  )

Base.show{S<:VariableState}(io::IO, ::Type{BasicDiscMuvParameter{S}}) = print(io, "BasicDiscMuvParameter")
Base.writemime{S<:VariableState}(io::IO, ::MIME"text/plain", t::Type{BasicDiscMuvParameter{S}}) = show(io, t)
