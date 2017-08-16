### Random Variable subtypes

## Abstract parameter

abstract type Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random} end

DiscreteParameter{F<:VariateForm} = Parameter{Discrete, F}
ContinuousParameter{F<:VariateForm} = Parameter{Continuous, F}
UnivariateParameter{S<:ValueSupport} = Parameter{S, Univariate}
MultivariateParameter{S<:ValueSupport} = Parameter{S, Multivariate}
MatrixvariateParameter{S<:ValueSupport} = Parameter{S, Matrixvariate}

ParameterVector{P<:Parameter} = Vector{P}

### Code generation of parameter fields

function codegen_setfield(parameter::Parameter, field::Symbol, f::Function)
  @gensym setfield
  quote
    function $setfield(_state::$(default_state_type(parameter)), _states::VariableStateVector)
      setfield!($parameter, $(QuoteNode(field)), $f(_state, _states))
    end
  end
end

function codegen_setfield(parameter::Parameter, field::Symbol, distribution::Symbol, f::Function)
  @gensym setfield
  quote
    function $setfield(_state::$(default_state_type(parameter)), _states::VariableStateVector)
      setfield!(_state, $(QuoteNode(field)), $(f)(getfield($parameter, $(QuoteNode(distribution))), _state.value))
    end
  end
end

macro codegen_closure(parameter, f)
  quote
    function (_state)
      $(esc(f))(_state, $(esc(parameter)).states)
    end
  end
end

function codegen_target_closure_via_distribution(parameter::Parameter, distribution::Symbol, f::Function, field::Symbol)
  @gensym target_closure_via_distribution
  quote
    function $target_closure_via_distribution(_state::$(default_state_type(parameter)))
      setfield!(_state, $(QuoteNode(field)), $(f)(getfield($parameter, $(QuoteNode(distribution))), _state.value))
    end
  end
end

function codegen_sumtarget_closure(
  parameter::Parameter, plfield::Symbol, ppfield::Symbol, stfield::Symbol, slfield::Symbol, spfield::Symbol
)
  body = []

  push!(body, :(getfield($parameter, $(QuoteNode(plfield)))(_state)))
  push!(body, :(getfield($parameter, $(QuoteNode(ppfield)))(_state)))
  push!(body, :(setfield!(
    _state,
    $(QuoteNode(stfield)),
    getfield(_state, $(QuoteNode(slfield)))+getfield(_state, $(QuoteNode(spfield)))))
  )

  @gensym sumtarget_closure

  quote
    function $sumtarget_closure(_state::$(default_state_type(parameter)))
      $(body...)
    end
  end
end

setpdf!(parameter::Parameter{S, F}, state::ParameterState{S, F}) where {S<:ValueSupport, F<:VariateForm} =
  parameter.setpdf(state, parameter.states)

setprior!(parameter::Parameter{S, F}, state::ParameterState{S, F}) where {S<:ValueSupport, F<:VariateForm} =
  parameter.setprior(state, parameter.states)

value_support{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = S
variate_form{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = F

function check_support(parameter::Parameter{S, F}, state::ParameterState{S, F}) where {S<:ValueSupport, F<:VariateForm}
  if value_support(parameter) != value_support(state)
    warn("Value support of parameter ($(value_support(parameter))) and of ($(value_support(state))) not in agreement")
  end

  if variate_form(parameter) != variate_form(state)
    error("Variate form of parameter ($(variate_form(parameter))) and of ($(variate_form(state))) not in agreement")
  end
end

dotshape(variable::Parameter) = "circle"
