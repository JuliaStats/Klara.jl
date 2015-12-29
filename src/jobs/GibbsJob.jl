### GibbsJob

type GibbsJob{S<:VariableState} <: MCJob
  model::GenericModel
  dpindices::Vector{Int} # Indices of dependent variables (parameters and transformations) in model.vertices
  dependent::Vector{Union{Parameter, Transformation}} # Points to model.vertices[dpindices] for faster access
  dpjobs::Dict{Symbol, BasicMCJob}
  range::BasicMCRange
  vstate::Vector{S} # Vector of variable states ordered according to variables in model.vertices
  output::Dict{Symbol, Union{VariableNState, VariableIOStream}} # Output of model's dependent variables
  ndp::Int # Number of dependent variables, i.e. length(dependent)
  count::Int # Current number of post-burnin iterations
  # task::Union{Task, Void}
  # resetplain!::Function
  iterate!::Function
  # reset!::Function
  # save!::Union{Function, Void}
  run!::Function
end
