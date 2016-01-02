### GibbsJob

type GibbsJob{S<:VariableState} <: MCJob
  model::GenericModel
  dpindices::Vector{Int} # Indices of dependent variables (parameters and transformations) in model.vertices
  dependent::Vector{Union{Parameter, Transformation}} # Points to model.vertices[dpindices] for faster access
  dpjobs::Vector{Union{BasicMCJob, Void}} # BasicMCJobs for parameters that will be sampled via Monte Carlo methods
  range::BasicMCRange
  vstate::Vector{S} # Vector of variable states ordered according to variables in model.vertices
  outopts::Dict # Options related to output
  output::Vector{Union{VariableNState, VariableIOStream, Void}} # Output of model's dependent variables
  ndp::Int # Number of dependent variables, i.e. length(dependent)
  count::Int # Current number of post-burnin iterations
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
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
    outopts::Dict,
    imperative::Bool, # If imperative=true then traverse graph imperatively, else declaratively via topological sorting
    plain::Bool,
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

    instance.ndp = length(dpindices)

    if !imperative
      instance.model = GenericModel(topological_sort_by_dfs(model), edges(model), is_directed(model))

      instance.dpindices = Array(Int, instance.ndp)
      for i in 1:instance.ndp
        instance.dpindices[i] = instance.model.ofkey[vertex_key(model.vertices[dpindices[i]])]
      end

      nvstate = length(vstate)
      instance.vstate = Array(S, nvstate)
      for i in 1:nvstate
        instance.vstate[instance.model.ofkey[vertex_key(model.vertices[i])]] = vstate[i]
      end

      instance.dpjobs = Array(Union{BasicMCJob, Void}, instance.ndp)
      for i in 1:instance.ndp
        instance.dpjobs[i] =
          if isa(dpjobs[i], BasicMCJob)
            BasicMCJob(
              instance.model,
              instance.model.vertices[instance.dpindices[i]],
              dpjobs[i].sampler,
              dpjobs[i].tuner,
              dpjobs[i].range,
              instance.vstate,
              dpjobs[i].outopts,
              dpjobs[i].plain,
              false
            )
          else
            nothing
          end
      end
    end

    instance.range = range
    instance.outopts = outopts
    instance.plain = plain

    instance.dependent = instance.model.vertices[instance.dpindices]
    for i in 1:instance.ndp
      instance.dependent[i].states = instance.vstate
    end

    # augment_outopts_gibbsjob!(outopts)
    # instance.output = initialize_output(instance.pstate, range.npoststeps, outopts)

    instance.count = 0

    # instance.iterate! = eval(codegen_iterate_gibbsjob(instance, outopts, plain))

    instance
  end
end

function checkin(job::GibbsJob)
  dpindices = find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), job.model.vertices)
  ndp = length(dpindices)

  @assert ndp > 0 "The model has neither parameters nor transformations, but at least one of them is required in a GibbsJob"

  ndpindices = length(job.dpindices)

  @assert ndpindices > 0 "Length of dpindices must be positive, got $ndpindices"

  if ndp == ndpindices
    if dpindices != job.dpindices
      error("Indices of parameters and transformations in model.vertices not same as dpindices")
    end
  elseif ndp < ndpindices
    error(
      "Number of parameters and transformations ($ndp) in model.vertices less than length of dpindices ($ndpindices)"
    )
  else # ndp > ndpindices
    warn(
      "Number of parameters and transformations ($ndp) in model.vertices greater than length of dpindices ($ndpindices)"
    )
    if in(job.dpindices, dpindices)
      error("Indices of parameters and transformations in model.vertices do not contain dpindices")
    end
  end

  ndpjobs = length(job.dpjobs)
  if ndpindices != ndpjobs
    error("Length of dpindices ($ndpindices) not equal to length of length of dpjobs ($ndpjobs)")
  else
    for i in 1:ndpindices
      if isa(job.dpjobs[i], BasicMCJob) && !isa(job.model.vertices[job.dpindices[i]], Parameter)
        error("BasicMCJob specified for model.vertices[$(job.dpindices[i])] variable, which is not a parameter")
      end
    end
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
