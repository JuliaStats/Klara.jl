### Generators are functions used for generating various specific models as instances of GenericModel

## likelihood_model represents a likelihood L(Vector{Parameter} | Vector{Data}, Vector{Hyperparameter})

function likelihood_model(vs::VariableVector; isdirected::Bool=true, isindexed::Bool=true)
  m = GenericModel(vs, Dependence[], isdirected=isdirected, isindexed=isindexed)

  pindex = find(v::Variable -> isa(v, Parameter), m.vertices)
  for i in pindex
    for j in setdiff(1:length(vs), pindex)
      add_edge!(m, m.vertices[j], m.vertices[i])
    end
  end

  return m
end

likelihood_model(v::Variable, isindexed::Bool=true) = GenericModel(Variable[v], Dependence[], isindexed=isindexed)
