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
  task::Union{Task, Void}
  close::Union{Function, Void}
  resetplain!::Union{Function, Void}
  reset!::Union{Function, Void}
  save!::Union{Function, Void}
  iterate!::Function
  run!::Function

  function GibbsJob(
    model::GenericModel,
    dpindex::Vector{Int},
    dpjob::Vector,
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

    if imperative
      instance.outopts = outopts
    else
      instance.model = GenericModel(topological_sort_by_dfs(model), edges(model), is_directed(model))

      idxs = indexes(instance.model)
      instance.dpindex = intersect(idxs, dpindex)
      instance.vstate = vstate[idxs]

      dpidxs = [findfirst(x -> x == instance.dpindex[i], dpindex) for i in 1:instance.ndp]

      instance.dpjob = Array(Union{BasicMCJob, Void}, instance.ndp)
      for i in 1:instance.ndp
        idpjob = dpjob[dpidxs[i]]
        instance.dpjob[i] =
          if isa(idpjob, BasicMCJob)
            BasicMCJob(
              instance.model,
              instance.dpindex[i],
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

      instance.outopts = outopts[dpidxs]
    end

    instance.range = range
    instance.imperative = imperative
    instance.plain = plain

    instance.dependent = instance.model.vertices[instance.dpindex]
    for i in 1:instance.ndp
      instance.dependent[i].states = instance.vstate
    end

    instance.dpstate = instance.vstate[instance.dpindex]

    instance.output = Array(Union{VariableNState, VariableIOStream, Void}, instance.ndp)
    for i in 1:instance.ndp
      if isa(instance.dependent[i], Parameter)
        augment_parameter_outopts!(instance.outopts[i])
      else
        augment_variable_outopts!(instance.outopts[i])
      end
      instance.output[i] = initialize_output(instance.dpstate[i], range.npoststeps, instance.outopts[i])
    end

    instance.count = 0

    instance.close = all(o -> o != VariableIOStream, instance.output) ? nothing : eval(codegen_close_gibbsjob(instance))

    nodpjob = all(j -> j == nothing, instance.dpjob)
    instance.resetplain! = nodpjob ? nothing : eval(codegen_resetplain_gibbsjob(instance))

    instance.save! = all(o -> o == nothing, instance.output) ? nothing : eval(codegen_save_gibbsjob(instance))

    instance.iterate! = eval(codegen_iterate_gibbsjob(instance))

    if plain
      instance.task = nothing
      instance.reset! = nodpjob ? nothing : instance.resetplain!
    else
      instance.task = Task(() -> initialize_task!(instance))
      instance.reset! = nodpjob ? nothing : eval(codegen_reset_task_gibbsjob(instance))
    end

    instance.run! = eval(codegen_run_gibbsjob(instance))

    instance
  end
end

function GibbsJob{S<:VariableState}(
  model::GenericModel,
  dpindex::Vector{Int},
  dpjob::Vector,
  range::BasicMCRange,
  vstate::Vector{S},
  outopts::Vector,
  imperative::Bool,
  plain::Bool,
  check::Bool
)
  job = isa(dpjob, Vector{Union{BasicMCJob, Void}}) ? dpjob : convert(Vector{Union{BasicMCJob, Void}}, dpjob)
  opts = isa(outopts, Vector{Dict{Symbol, Any}}) ? outopts : convert(Vector{Dict{Symbol, Any}}, outopts)
  GibbsJob{S}(model, dpindex, job, range, vstate, opts, imperative, plain, check)
end

GibbsJob{S<:VariableState}(
  model::GenericModel,
  dpjob::Vector,
  range::BasicMCRange,
  v0::Vector{S};
  dpindex::Vector{Int}=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Vector=[Dict(:destination=>:nstate, :monitor=>[:value]) for i in 1:length(dpindex)],
  imperative::Bool=true,
  plain::Bool=true,
  check::Bool=false
) =
  GibbsJob{S}(model, dpindex, dpjob, range, v0, outopts, imperative, plain, check)

function GibbsJob{S<:VariableState}(
  model::GenericModel,
  dpjob::Dict,
  range::BasicMCRange,
  v0::Dict{Symbol, S};
  dpindex::Vector{Int}=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Dict=Dict([(k, Dict(:destination=>:nstate, :monitor=>[:value])) for k in keys(model.vertices[dpindex])]),
  imperative::Bool=true,
  plain::Bool=true,
  check::Bool=false
)
  ndpindex = length(dpindex)

  jobs = Array(Union{BasicMCJob, Void}, ndpindex)
  opts = Array(Dict, ndpindex)

  for i in 1:ndpindex
    key = vertex_key(model.vertices[dpindex[i]])
    jobs[i] = haskey(dpjob, key) ? dpjob[key] : nothing
    opts[i] = haskey(outopts, key) ? outopts[key] : Dict(:destination=>:nstate, :monitor=>[:value])
  end

  vstate = Array(S, length(v0))
  for (k, v) in v0
    vstate[model.ofkey[k]] = v
  end

  GibbsJob(model, dpindex, jobs, range, vstate, opts, imperative, plain, check)
end

function codegen_close_gibbsjob(job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.output[j], VariableIOStream)
      push!(body, :($(job).output[$j].close()))
    end
  end

  @gensym close_gibbsjob

  quote
    function $close_gibbsjob()
      $(body...)
    end
  end
end

function codegen_resetplain_gibbsjob(job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.dpjob[j], BasicMCJob)
      if job.dpjob[j].resetpstate
        push!(body, :(reset($(job).dpjob[$j], rand($(job).dependent[$j].prior))))
      else
        push!(body, :(reset($(job).dpjob[$j])))
      end
    end
  end

  @gensym resetplain_gibbsjob

  quote
    function $resetplain_gibbsjob()
      $(body...)
    end
  end
end

function codegen_reset_task_gibbsjob(job::GibbsJob)
  @gensym reset_task_gibbsjob
  Expr(:function, Expr(:call, reset_task_gibbsjob), Expr(:block, :($(job).task.storage[:reset]())))
end

Base.reset(job::GibbsJob) = job.reset!()

function codegen_save_gibbsjob(job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.output[j], VariableNState)
      push!(body, :($(job).output[$j].copy($(job).dpstate[$j], _i)))
    elseif isa(job.output[j], VariableIOStream)
      push!(body, :($(job).output[$j].write($(job).dpstate[$j])))
      if job.outopts[j][:flush]
        push!(body, :($(job).outopts[$j].flush()))
      end
    else
      error("To save output, :destination must be set to :nstate or :iostream, got $(job.outopts[j][:destination])")
    end
  end

  @gensym save_gibbsjob

  quote
    function $save_gibbsjob(_i::Int)
      $(body...)
    end
  end
end

function codegen_iterate_gibbsjob(job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.dependent[j], Parameter)
      if isa(job.dpjob[j], BasicMCJob)
        push!(body, :(run($(job).dpjob[$j])))
      else
        push!(body, :($(job).dependent[$j].setpdf($(job).dpstate[$j])))
        push!(body, :($(job).dpstate[$j].value = rand($(job).dependent[$j].pdf)))
      end
    else
      push!(body, :($(job).dependent[$j].transform!($(job).dpstate[$j])))
    end
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym iterate_gibbsjob

  quote
    function $iterate_gibbsjob()
      $(body...)
    end
  end
end

function codegen_run_gibbsjob(job::GibbsJob)
  result::Expr
  ifforbody = []
  forbody = []
  body = []

  if job.task == nothing
    push!(forbody, :($(job).iterate!()))
  else
    push!(forbody, :(consume($(job).task)))
  end

  push!(ifforbody, :($(job).count+=1))
  if job.save! != nothing
    push!(ifforbody, :($(job).save!($(job).count)))
  end

  push!(forbody, Expr(:if, :(in(i, $(job).range.postrange)), Expr(:block, ifforbody...)))

  if job.reset! != nothing
    push!(forbody, :($(job).reset!()))
  end

  push!(body, Expr(:for, :(i = 1:$(job).range.nsteps), Expr(:block, forbody...)))

  if isa(job.output, VariableIOStream)
    push!(body, :($(job).output.close()))
  end

  if job.close != nothing
    push!(body, :($(job).close()))
  end

  @gensym run_gibbsjob

  result = quote
    function $run_gibbsjob()
      $(body...)
    end
  end

  result
end

function Base.run(job::GibbsJob)
  for i in 1:job.range.nsteps
    iterate!(job)

    if in(i, job.range.postrange)
      job.count+=1
      save!(job)
    end

    reset!(job)
  end

  for j in 1:job.ndp
    if isa(job.output[j], VariableIOStream)
      job.output[j].close()
    end
  end
end

Base.run(job::GibbsJob) = job.run!()

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

dpkeys(job::GibbsJob) = Symbol[dp.key for dp in job.dependent]

output(job::GibbsJob) = job.output

Dict(job::GibbsJob, field::Symbol=:output) = Dict(zip(dpkeys(job), getfield(job, field)))

function Base.show(io::IO, job::GibbsJob)
  ndptransforms = num_dptransforms(job)
  ndpviamcmc = num_randdp_viamcmc(job)
  ndpviadistribution = job.ndp-ndptransforms-ndpviamcmc

  isimperative = job.imperative ? "imperative graph traversal" : "declarative graph traversal via topologically sorted nodes"
  isplain = job.plain ? "job flow not controlled by tasks" : "job flow controlled by tasks"

  println(io, "GibbsJob:")
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
