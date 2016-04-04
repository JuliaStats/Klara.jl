### Random Variable subtypes

## Abstract parameter

abstract Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random}

typealias ParameterVector{P<:Parameter} Vector{P}

setpdf!(parameter::Parameter, state::ParameterState) = parameter.setpdf(state, parameter.states)
setprior!(parameter::Parameter, state::ParameterState) = parameter.setprior(state, parameter.states)

value_support{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = S
variate_form{S<:ValueSupport, F<:VariateForm}(::Type{Parameter{S, F}}) = F

function check_support(parameter::Parameter, state::ParameterState)
  if value_support(parameter) != value_support(state)
    warn("Value support of parameter ($(value_support(parameter))) and of ($(value_support(state))) not in agreement")
  end

  if variate_form(parameter) != variate_form(state)
    error("Variate form of parameter ($(variate_form(parameter))) and of ($(variate_form(state))) not in agreement")
  end
end

dotshape(variable::Parameter) = "circle"
