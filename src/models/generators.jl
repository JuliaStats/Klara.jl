### Generators are functions used for generating various specific models as instances of GenericModel

## likelihood_model represents a likelihood L(Vector{Parameter} | Vector{Data}, Vector{Hyperparameter})

function likelihood_model{P<:Parameter}(
  p::Vector{P};
  data::Vector{Data}=Array(Data, 0),
  hyperparameters::Vector{Hyperparameter}=Array(Hyperparameter, 0),
  isdirected::Bool=true,
  isindexed::Bool=true
)
  variables = [p; data; hyperparameters]

  m = GenericModel(variables, Dependence[], isdirected=isdirected, isindexed=isindexed)

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
  isdirected::Bool=true,
  isindexed::Bool=true
) =
  likelihood_model(Parameter[p], data=data, hyperparameters=hyperparameters, isdirected=isdirected, isindexed=isindexed)
