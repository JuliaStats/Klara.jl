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
  verbose::Bool
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
    verbose::Bool,
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
      idxs = [
        model.ofkey[k]
        for k in keys(topological_sort_by_dfs(GenericModel(vertices(model), edges(model), is_directed(model))))
      ]
      instance.dpindex = intersect(idxs, dpindex)

      dpidxs = map(x -> findfirst(dpindex, x), instance.dpindex)

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
    instance.verbose = verbose

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

    instance.close = all(o -> o != VariableIOStream, instance.output) ? nothing : eval(codegen(:close, instance))

    nodpjob = all(j -> j == nothing, instance.dpjob)
    instance.resetplain! = nodpjob ? nothing : eval(codegen(:resetplain, instance))

    instance.save! = all(o -> o == nothing, instance.output) ? nothing : eval(codegen(:save, instance))

    instance.iterate! = eval(codegen(:iterate, instance))

    if plain
      instance.task = nothing
      instance.reset! = nodpjob ? nothing : instance.resetplain!
    else
      instance.task = Task(() -> initialize_task!(instance))
      instance.reset! = nodpjob ? nothing : eval(codegen(:resettask, instance))
    end

    instance.run! = eval(codegen(:run, instance))

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
  verbose::Bool,
  check::Bool
)
  job = isa(dpjob, Vector{Union{BasicMCJob, Void}}) ? dpjob : convert(Vector{Union{BasicMCJob, Void}}, dpjob)
  opts = isa(outopts, Vector{Dict{Symbol, Any}}) ? outopts : convert(Vector{Dict{Symbol, Any}}, outopts)
  GibbsJob{S}(model, dpindex, job, range, vstate, opts, imperative, plain, verbose, check)
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
  verbose::Bool=false,
  check::Bool=false
) =
  GibbsJob{S}(model, dpindex, dpjob, range, v0, outopts, imperative, plain, verbose, check)

function GibbsJob{S<:VariableState}(
  model::GenericModel,
  dpjob::Dict,
  range::BasicMCRange,
  v0::Dict{Symbol, S};
  dpindex::Vector{Int}=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Dict=Dict([(k, Dict(:destination=>:nstate, :monitor=>[:value])) for k in keys(model.vertices[dpindex])]),
  imperative::Bool=true,
  plain::Bool=true,
  verbose::Bool=false,
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

  GibbsJob(model, dpindex, jobs, range, vstate, opts, imperative, plain, verbose, check)
end

function GibbsJob(
  model::GenericModel,
  dpjob::Vector,
  range::BasicMCRange,
  v0::Vector;
  dpindex::Vector{Int}=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Vector=[Dict(:destination=>:nstate, :monitor=>[:value]) for i in 1:length(dpindex)],
  imperative::Bool=true,
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
)
  vstate = default_state(model.vertices, v0, outopts, dpindex)
  GibbsJob(model, dpindex, dpjob, range, vstate, outopts, imperative, plain, verbose, check)
end

function GibbsJob(
  model::GenericModel,
  dpjob::Dict,
  range::BasicMCRange,
  v0::Dict;
  dpindex::Vector{Int}=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Dict=Dict([(k, Dict(:destination=>:nstate, :monitor=>[:value])) for k in keys(model.vertices[dpindex])]),
  imperative::Bool=true,
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
)
  vstate = Dict{Symbol, VariableState}()
  for (k, v) in v0
    vstate[k] = default_state(model[k], v, get!(outopts, k, Dict()))
  end

  GibbsJob(
    model,
    dpjob,
    range,
    vstate,
    dpindex=dpindex,
    outopts=outopts,
    imperative=imperative,
    plain=plain,
    verbose=verbose,
    check=check
  )
end

codegen(f::Symbol, job::GibbsJob) = codegen(Val{f}, job)

function codegen(::Type{Val{:close}}, job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.output[j], VariableIOStream)
      push!(body, :(close(_job.output[$j])))
    end
  end

  @gensym _close

  quote
    function $_close(_job::GibbsJob)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:resetplain}}, job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.dpjob[j], BasicMCJob)
      if job.dpjob[j].resetpstate
        push!(body, :(reset(_job.dpjob[$j], rand(_job.dependent[$j].prior))))
      else
        push!(body, :(reset(_job.dpjob[$j])))
      end
    end
  end

  @gensym _resetplain

  quote
    function $_resetplain(_job::GibbsJob)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:resettask}}, job::GibbsJob)
  @gensym _reset
  Expr(:function, Expr(:call, _reset, :(_job::GibbsJob)), Expr(:block, :(_job.task.storage[:reset](_job))))
end

function codegen(::Type{Val{:save}}, job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.output[j], VariableNState)
      push!(body, :(copy!(_job.output[$j], _job.dpstate[$j], _i)))
    elseif isa(job.output[j], VariableIOStream)
      push!(body, :(write(_job.output[$j], _job.dpstate[$j])))
      if job.outopts[j][:flush]
        push!(body, :(flush(_job.outopts[$j])))
      end
    else
      error("To save output, :destination must be set to :nstate or :iostream, got $(job.outopts[j][:destination])")
    end
  end

  @gensym _save

  quote
    function $_save(_job::GibbsJob, _i::Int)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:iterate}}, job::GibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.dependent[j], Parameter)
      if isa(job.dpjob[j], BasicMCJob)
        push!(body, :(run(_job.dpjob[$j])))
      else
        push!(body, :(_job.dependent[$j].setpdf(_job.dpstate[$j])))
        push!(body, :(_job.dpstate[$j].value = rand(_job.dependent[$j].pdf)))
      end
    else
      push!(body, :(_job.dependent[$j].transform!(_job.dpstate[$j])))
    end
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym _iterate

  quote
    function $_iterate(_job::GibbsJob)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:run}}, job::GibbsJob)
  result::Expr
  ifforbody = []
  forbody = []
  body = []

  if job.verbose
    fmt_iter = format_iteration(ndigits(job.range.nsteps))
    push!(forbody, :(println("Iteration ", $(fmt_iter)(i), " of ", _job.range.nsteps)))
  end

  if job.task == nothing
    push!(forbody, :(iterate(_job)))
  else
    push!(forbody, :(consume(_job.task)))
  end

  push!(ifforbody, :(_job.count+=1))
  if job.save! != nothing
    push!(ifforbody, :(save(_job, _job.count)))
  end

  push!(forbody, Expr(:if, :(in(i, _job.range.postrange)), Expr(:block, ifforbody...)))

  if job.reset! != nothing
    push!(forbody, :(reset(_job)))
  end

  push!(body, Expr(:for, :(i = 1:_job.range.nsteps), Expr(:block, forbody...)))

  if isa(job.output, VariableIOStream)
    push!(body, :(close(_job.output)))
  end

  if job.close != nothing
    push!(body, :(close(_job)))
  end

  @gensym _run

  result = quote
    function $_run(_job::GibbsJob)
      $(body...)
    end
  end

  result
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

dpkeys(job::GibbsJob) = Symbol[dp.key for dp in job.dependent]

output(job::GibbsJob) = job.output

Dict(job::GibbsJob, field::Symbol=:output) = Dict(zip(dpkeys(job), getfield(job, field)))

function Base.show(io::IO, job::GibbsJob)
  ndptransforms = num_dptransforms(job)
  ndpviamcmc = num_randdp_viamcmc(job)
  ndpviadistribution = job.ndp-ndptransforms-ndpviamcmc

  isimperative = job.imperative ? "imperative graph traversal" : "declarative graph traversal via topologically sorted nodes"
  isplain = job.plain ? "job flow not controlled by tasks" : "job flow controlled by tasks"

  indentation = "  "

  println(io, "GibbsJob:")
  print(io, string(
    indentation,
    "$(num_dp(job)) dependent variables: ",
    "$ndptransforms transformations, ",
    "$ndpviadistribution sampled from their distribution, ",
    "$ndpviamcmc sampled via MCMC-within-Gibbs"
  ))
  print(io, "\n"*indentation)
  show(io, job.model)
  print(io, "\n"*indentation)
  show(io, job.range)
  println(io, "\n"*indentation*"plain = $(job.plain) ($isplain)")
  print(io, indentation*"imperative = $(job.imperative) ($isimperative)")
end

Base.writemime(io::IO, ::MIME"text/plain", job::GibbsJob) = show(io, job)

function job2dot(stream::IOStream, job::GibbsJob)
  graphkeyword, edgesign = is_directed(job.model) ? ("digraph", "->") : ("graph", "--")
  dotindentation, dotspacing = "  ", " "

  write(stream, "$graphkeyword GibbsJob {\n")

  invdpindex = Dict{Symbol, Int}(zip(keys(job.model.vertices[job.dpindex]), 1:job.ndp))

  for v in vertices(job.model)
    vstring = string(dotindentation, v.key, dotspacing, "[shape=", dotshape(v))

    if isa(v, Parameter) || isa(v, Transformation)
      vstring *= ","*dotspacing*"peripheries=2"

      i = invdpindex[v.key]

      if job.outopts[i][:destination] != :none
        vstring *= ","*dotspacing*"label=<<u>$(v.key)</u>>"
      end

      if isa(v, Parameter) && (job.dpjob[i] != nothing)
        vstring *= ","*dotspacing*"style=diagonals"
      end
    end

    write(stream, vstring*"]\n")
  end

  for d in edges(job.model)
    write(stream, string(dotindentation, d.source.key, dotspacing, edgesign, dotspacing, d.target.key, "\n"))
  end

  write(stream, "}\n")
end

function job2dot(filename::AbstractString, job::GibbsJob, mode::AbstractString="w")
  stream = open(filename, mode)
  job2dot(stream, job)
  close(stream)
end
