### Random Variable subtypes

## Abstract parameter

abstract Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random}

typealias DiscreteParameter{F<:VariateForm} Parameter{Discrete, F}
typealias ContinuousParameter{F<:VariateForm} Parameter{Continuous, F}

typealias UnivariateParameter{S<:ValueSupport} Parameter{S, Univariate}
typealias MultivariateParameter{S<:ValueSupport} Parameter{S, Multivariate}
typealias MatrixvariateParameter{S<:ValueSupport} Parameter{S, Matrixvariate}

typealias ParameterVector{P<:Parameter} Vector{P}

### Code generation of parameter fields

function codegen_setfield(parameter::Parameter, field::Symbol, f::Function)
  @gensym setfield
  quote
    function $setfield(_state::$(default_state_type(parameter)), _states::VariableStateVector)
      setfield!($parameter, $(QuoteNode(field)), $f(_state, _states))
    end
  end
end

function codegen_closure(parameter::Parameter, f::Function)
  @gensym closure
  quote
    function $closure(_state::$(default_state_type(parameter)))
      $(f)(_state, $(parameter).states)
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

function codegen_uptotarget_closures(parameter::Parameter, fields::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:length(fields)
    f = fields[i]
    push!(body, :(getfield($parameter, $(QuoteNode(f)))(_state)))
  end

  @gensym uptotarget_closures

  quote
    function $uptotarget_closures(_state::$(default_state_type(parameter)))
      $(body...)
    end
  end
end

setpdf!{S<:ValueSupport, F<:VariateForm}(parameter::Parameter{S, F}, state::ParameterState{S, F}) =
  parameter.setpdf(state, parameter.states)

setprior!{S<:ValueSupport, F<:VariateForm}(parameter::Parameter{S, F}, state::ParameterState{S, F}) =
  parameter.setprior(state, parameter.states)

value_support{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = S
variate_form{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = F

function check_support{S<:ValueSupport, F<:VariateForm}(parameter::Parameter{S, F}, state::ParameterState{S, F})
  if value_support(parameter) != value_support(state)
    warn("Value support of parameter ($(value_support(parameter))) and of ($(value_support(state))) not in agreement")
  end

  if variate_form(parameter) != variate_form(state)
    error("Variate form of parameter ($(variate_form(parameter))) and of ($(variate_form(state))) not in agreement")
  end
end

dotshape(variable::Parameter) = "circle"
