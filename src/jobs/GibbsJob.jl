### GibbsJob

type GibbsJob{S<:VariableState} <: MCJob
  model::GenericModel
  dpindices::Vector{Int} # Indices of dependent variables (parameters and transformations) in model.vertices
  dependent::Vector{Union{Parameter, Transformation}} # Points to model.vertices[dpindices] for faster access
  dpjobs::Dict{Symbol, BasicMCJob} # BasicMCJobs for parameters (in dependent) that will be sampled via Monte Carlo methods
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

  function GibbsJob(
    model::GenericModel,
    dpindices::Vector{Int},
    dpjobs::Dict{Symbol, BasicMCJob},
    range::BasicMCRange,
    vstate::Vector{S},
    outopts::Dict, # Options related to output
    imperative::Bool, # If imperative=true then traverse graph imperatively, else declaratively via topological sorting
    plain::Bool, # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
    check::Bool
  )
    instance = new()

    instance.model = model
    instance.dpindices = dpindices
    instance.dpjobs = dpjobs
    instance.vstate = vstate

    if check
      checkin(instance)
    end

    if !imperative
      instance.model = GenericModel(topological_sort_by_dfs(model), model.edges, model.is_directed)

      instance.dpindices = Array(Int, length(dpindices))
      for i in 1:length(dpindices)
        instance.dpindices[i] = instance.model.ofkey[vertex_key(model.vertices[dpindices[i]])]
      end

      instance.vstate = Array(S, length(vstate))
      for i in 1:length(vstate)
        instance.vstate[instance.model.ofkey[vertex_key(model.vertices[i])]] = vstate[i]
      end
    end

    # instance.range = range
    #
    # if !imperative
    #   # reset model, dpindices, vstate, dpjobs
    # end
    #
    # instance.dependent = instance.model.vertices[instance.dpindices]
    # instance.ndp = length(instance.dependent)
    # for i in 1:instance.ndp
    #   instance.dependent[i].states = instance.vstate
    # end
    #
    # for v in values(instance.dpjobs)
    #   v.vstate = instance.vstate
    #   v.parameter.states = v.vstate
    # end
    #
    # augment_outopts_gibbsjob!(outopts)
    # instance.output = initialize_output(instance.pstate, range.npoststeps, outopts)
    #
    # instance.count = 0
    #
    # instance.iterate! = eval(codegen_iterate_gibbsjob(instance, outopts, plain))

    instance
  end
end

function checkin(job::GibbsJob)
  dpindices = find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), job.model.vertices)
  ndp = length(dpindices)

  @assert ndp > 0 "The model has neither parameters nor transformations, but at least one of them is required in a GibbsJob"

  ndpindices = length(job.dpindices)

  @assert ndpindices > 0 "Length of job.dpindices must be positive, got $ndpindices"

  if ndp == ndpindices
    if dpindices != job.dpindices
      error("Indices of parameters and transformations in model.vertices not same as job.dpindices")
    end
  elseif ndp < ndpindices
    error(
      "Number of parameters and transformations ($ndp) in model.vertices less than length of job.dpindices ($ndpindices)"
    )
  else # ndp > ndpindices
    warn(
      "Number of parameters and transformations ($ndp) in model.vertices greater than length of job.dpindices ($ndpindices)"
    )
    if in(job.dpindices, dpindices)
      error("Indices of parameters and transformations in model.vertices do not contain job.dpindices ")
    end
  end

  if !issubset(keys(job.dpjobs), keys(job.model))
    warn("Keys of job.dpjobs not a subset of keys of job.model")
  end

  nv = num_vertices(job.model)
  nvstate = length(job.vstate)

  if nv != nvstate
    warn("Number of variables ($nv) not equal to number of variable states ($nvstate)")
  end

  for i in ndpindices
    if isa(job.model.vertices[i], Parameter)
      if !isa(job.vstate[i], ParameterState)
        error("The parameter's state must be saved in a ParameterState subtype, got $(typeof(job.vstate[i])) state type")
      else
        check_support(job.model.vertices[i], job.vstate[i])
      end
    end
  end
end
