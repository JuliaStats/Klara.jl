### Random Variable subtypes

## Abstract parameter

abstract type Parameter{S<:ValueSupport, F<:VariateForm} <: Variable{Random} end

DiscreteParameter{F<:VariateForm} = Parameter{Discrete, F}
ContinuousParameter{F<:VariateForm} = Parameter{Continuous, F}
UnivariateParameter{S<:ValueSupport} = Parameter{S, Univariate}
MultivariateParameter{S<:ValueSupport} = Parameter{S, Multivariate}
MatrixvariateParameter{S<:ValueSupport} = Parameter{S, Matrixvariate}

ParameterVector{P<:Parameter} = Vector{P}

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
