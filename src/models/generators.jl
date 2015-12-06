### Generators are functions used for generating various specific models as instances of GenericModel

## likelihood_model represents a likelihood L(Vector{Parameter} | Vector{Data}, Vector{Hyperparameter})

function likelihood_model{P<:Parameter}(
  p::Vector{P};
  data::Vector{Data}=Array(Data, 0),
  hyperparameters::Vector{Hyperparameter}=Array(Hyperparameter, 0),
  is_directed::Bool=true,
  is_indexed::Bool=true
)
  m = GenericModel(is_directed)
  variables = [p; data; hyperparameters]

  if !is_indexed
    for i in 1:length(variables)
      variables[i].index = i
    end
  end

  for v in variables
    add_vertex!(m, v)
    m.indexof[v] = v.index
  end

  for t in p
    for s in variables[(length(p)+1):end]
      add_edge!(m, s, t)
    end
  end

  return m
end

## single_parameter_likelihood_model represents a likelihood L(Parameter | Vector{Data}, Vector{Hyperparameter})

single_parameter_likelihood_model(
  p::Parameter;
  data::Vector{Data}=Array(Data, 0),
  hyperparameters::Vector{Hyperparameter}=Array(Hyperparameter, 0),
  is_directed::Bool=true,
  is_indexed::Bool=true
) =
  likelihood_model(Parameter[p], data=data, hyperparameters=hyperparameters, is_directed=is_directed, is_indexed=is_indexed)
