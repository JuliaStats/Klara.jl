### GibbsJob

type GibbsJob{S<:VariableState} <: MCJob
  model::GenericModel
  dpindex::Vector{Int} # Indices of dependent variables (parameters and transformations) in model.vertices
  dependent::Vector{Union{Parameter, Transformation}} # Points to model.vertices[dpindex] for faster access
  dpjob::Vector{Union{BasicMCJob, Void}} # BasicMCJobs for parameters that will be sampled via Monte Carlo methods
  range::BasicMCRange
  vstate::Vector{S} # Vector of variable states ordered according to variables in model.vertices
  dpstate::Vector{VariableState} # Points to vstate[dpindex] for faster access
  outopts::Vector # Options related to output
  output::Vector{Union{VariableNState, VariableIOStream, Void}} # Output of model's dependent variables
  ndp::Int # Number of dependent variables, i.e. length(dependent)
  count::Int # Current number of post-burnin iterations
  imperative::Bool # If imperative=true then traverse graph imperatively, else declaratively via topological sorting
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  # task::Union{Task, Void}
  # resetplain!::Function
  # iterate!::Function
  # reset!::Function
  # save!::Union{Function, Void}
  # run!::Function

  function GibbsJob(
    model::GenericModel,
    dpindex::Vector{Int},
    dpjob::Vector{Union{BasicMCJob, Void}},
    range::BasicMCRange,
    vstate::Vector{S},
    outopts::Vector,
    imperative::Bool,
    plain::Bool,
    check::Bool
  )
    instance = new()

    instance.model = model
    instance.dpindex = dpindex
    instance.dpjob = dpjob
    instance.vstate = vstate

    if check
      checkin(instance)
    end

    instance.ndp = length(dpindex)

    if !imperative
      instance.model = GenericModel(topological_sort_by_dfs(model), edges(model), is_directed(model))

      instance.dpindex = Array(Int, instance.ndp)
      nvstate = length(vstate)
      instance.vstate = Array(S, nvstate)

      for i in 1:instance.ndp
        instance.dpindex[i] = instance.model.ofkey[vertex_key(model.vertices[dpindex[i]])]
        instance.vstate[instance.dpindex[i]] = vstate[dpindex[i]]
      end

      for i in setdiff(1:nvstate, dpindex)
        instance.vstate[instance.model.ofkey[vertex_key(model.vertices[i])]] = vstate[i]
      end
    end

    instance.range = range
    instance.outopts = outopts
    instance.imperative = imperative
    instance.plain = plain

    instance.dependent = instance.model.vertices[instance.dpindex]
    for i in 1:instance.ndp
      instance.dependent[i].states = instance.vstate
    end

    if !imperative
      instance.dpjob = Array(Union{BasicMCJob, Void}, instance.ndp)
      idpjob::BasicMCJob
      for i in 1:instance.ndp
        idpjob = dpjob[i]
        instance.dpjob[i] =
          if isa(idpjob, BasicMCJob)
            BasicMCJob(
              instance.model,
              instance.dependent[i],
              idpjob.sampler,
              idpjob.tuner,
              idpjob.range,
              instance.vstate,
              idpjob.outopts,
              idpjob.plain,
              false
            )
          else
            nothing
          end
      end
    end

    instance.dpstate = instance.vstate[instance.dpindex]

    instance.output = Array(Union{VariableNState, VariableIOStream, Void}, instance.ndp)
    for i in 1:instance.ndp
      if isa(instance.dependent[i], Parameter)
        augment_parameter_outopts!(outopts[i])
      else
        augment_variable_outopts!(outopts[i])
      end
      instance.output[i] = initialize_output(instance.dpstate[i], range.npoststeps, outopts[i])
    end

    instance.count = 0

    # instance.iterate! = eval(codegen_iterate_gibbsjob(instance, outopts, plain))

    instance
  end
end

GibbsJob{S<:VariableState}(
  model::GenericModel,
  dpindex::Vector{Int},
  dpjob::Vector{Union{BasicMCJob, Void}},
  range::BasicMCRange,
  vstate::Vector{S},
  outopts::Vector,
  imperative::Bool,
  plain::Bool,
  check::Bool
) =
  GibbsJob{S}(model, dpindex, dpjob, range, vstate, outopts, imperative, plain, check)

function iterate!(job::GibbsJob)
  for j in 1:job.ndp
    if isa(job.dependent[i], Parameter)
      if isa(job.dpjob[i], BasicMCJob)
        run(job.dpjob[i])
        reset(job, job.dpstate[i])
      else
        job.dependent[i].setpdf(job.dpstate[i])
        job.dpstate[i] = rand(job.pdf)
      end
    else
      job.dependent[i].transform!(job.dpstate[i])
    end
  end
end

function Base.run(job::GibbsJob)
  for i in 1:job.range.nsteps
    iterate!(job)

    #if in(i, job.range.postrange)
    #  job.count+=1
    #  job.save!(job.count)
    #end
  end

  #job.close()

  #job.output
end

function checkin(job::GibbsJob)
  dpindex = find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), job.model.vertices)
  ndp = length(dpindex)

  @assert ndp > 0 "The model has neither parameters nor transformations, but at least one of them is required in a GibbsJob"

  ndpindex = length(job.dpindex)

  @assert ndpindex > 0 "Length of dpindex must be positive, got $ndpindex"

  if ndp == ndpindex
    if dpindex != job.dpindex
      error("Indices of parameters and transformations in model.vertices not same as dpindex")
    end
  elseif ndp < ndpindex
    error("Number of parameters and transformations ($ndp) in model.vertices less than length of dpindex ($ndpindex)")
  else # ndp > ndpindex
    warn("Number of parameters and transformations ($ndp) in model.vertices greater than length of dpindex ($ndpindex)")
    if in(job.dpindex, dpindex)
      error("Indices of parameters and transformations in model.vertices do not contain dpindex")
    end
  end

  ndpjob = length(job.dpjob)
  if ndpindex != ndpjob
    error("Length of dpindex ($ndpindex) not equal to length of length of dpjob ($ndpjob)")
  else
    for i in 1:ndpindex
      if isa(job.dpjob[i], BasicMCJob) && !isa(job.model.vertices[job.dpindex[i]], Parameter)
        error("BasicMCJob specified for model.vertices[$(job.dpindex[i])] variable, which is not a parameter")
      end
    end
  end

  nv = num_vertices(job.model)
  nvstate = length(job.vstate)

  if nv != nvstate
    warn("Number of variables ($nv) not equal to number of variable states ($nvstate)")
  end

  for i in ndpindex
    if isa(job.model.vertices[i], Parameter)
      if !isa(job.vstate[i], ParameterState)
        error("The parameter's state must be saved in a ParameterState subtype, got $(typeof(job.vstate[i])) state type")
      else
        check_support(job.model.vertices[i], job.vstate[i])
      end
    end
  end
end

num_dp(job::GibbsJob) = job.ndp
num_dptransforms(job::GibbsJob) = count(v -> isa(v, Transformation), job.dependent)
num_randdp(job::GibbsJob) = count(v -> isa(v, Parameter), job.dependent)
num_randdp_viamcmc(job::GibbsJob) = count(j -> j != nothing, job.dpjob)
num_radndp_viadistribution(job::GibbsJob) = num_dp(job)-num_dptransforms(job)-num_randdp_viamcmc(job)

function Base.show(io::IO, job::GibbsJob)
  ndptransforms = num_dptransforms(job)
  ndpviamcmc = num_randdp_viamcmc(job)
  ndpviadistribution = job.ndp-ndptransforms-ndpviamcmc

  isimperative = job.imperative ? "imperative graph traversal" : "declarative graph traversal (topologically sorted nodes)"
  isplain = job.plain ? "job flow not controlled by tasks" : "job flow controlled by tasks"

  println(io, "BasicMCJob:")
  print(io, string(
    "  $(num_dp(job)) dependent variables: ",
    "$ndptransforms transformations, ",
    "$ndpviadistribution sampled from their distribution, ",
    "$ndpviamcmc sampled via MCMC-within-Gibbs"
  ))
  print(io, "\n  ")
  show(io, job.model)
  print(io, "\n  ")
  show(io, job.range)
  println(io, "\n  plain = $(job.plain) ($isplain)")
  print(io, "  imperative = $(job.imperative) ($isimperative)")
end

Base.writemime(io::IO, ::MIME"text/plain", job::GibbsJob) = show(io, job)
