### BasicGibbsJob

type BasicGibbsJob <: GibbsJob
  model::GenericModel
  dpindex::IntegerVector # Indices of dependent variables (parameters and transformations) in model.vertices
  dependent::Vector{Union{Parameter, Transformation}} # Points to model.vertices[dpindex] for faster access
  dpjob::Vector{Union{BasicMCJob, Void}} # BasicMCJobs for parameters that will be sampled via Monte Carlo methods
  range::BasicMCRange
  vstate::VariableStateVector # Vector of variable states ordered according to variables in model.vertices
  dpstate::VariableStateVector # Points to vstate[dpindex] for faster access
  outopts::Vector # Options related to output
  output::Vector{Union{VariableNState, VariableIOStream, Void}} # Output of model's dependent variables
  ndp::Integer # Number of dependent variables, i.e. length(dependent)
  count::Integer # Current number of post-burnin iterations
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  task::Union{Task, Void}
  verbose::Bool
  close::Union{Function, Void}
  resetplain!::Union{Function, Void}
  reset!::Union{Function, Void}
  save!::Union{Function, Void}
  iterate!::Function
  run!::Function

  function BasicGibbsJob(
    model::GenericModel,
    dpindex::IntegerVector,
    dpjob::Vector,
    range::BasicMCRange,
    vstate::VariableStateVector,
    outopts::Vector,
    plain::Bool,
    verbose::Bool,
    check::Bool
  )
    instance = new()

    instance.model = model
    instance.dpindex = dpindex
    instance.dpjob = isa(dpjob, Vector{Union{BasicMCJob, Void}}) ? dpjob : convert(Vector{Union{BasicMCJob, Void}}, dpjob)
    instance.vstate = vstate

    if check
      checkin(instance)
    end

    instance.ndp = length(dpindex)

    instance.outopts = isa(outopts, Vector{Dict{Symbol, Any}}) ? outopts : convert(Vector{Dict{Symbol, Any}}, outopts)

    instance.range = range
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

BasicGibbsJob(
  model::GenericModel,
  dpjob::Vector,
  range::BasicMCRange,
  v0::VariableStateVector;
  dpindex::IntegerVector=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Vector=[Dict(:destination=>:nstate, :monitor=>[:value]) for i in 1:length(dpindex)],
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
) =
  BasicGibbsJob(model, dpindex, dpjob, range, v0, outopts, plain, verbose, check)

function BasicGibbsJob{S<:VariableState}(
  model::GenericModel,
  dpjob::Dict,
  range::BasicMCRange,
  v0::Dict{Symbol, S};
  dpindex::IntegerVector=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Dict=Dict([(k, Dict(:destination=>:nstate, :monitor=>[:value])) for k in keys(model.vertices[dpindex])]),
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

  BasicGibbsJob(model, dpindex, jobs, range, vstate, opts, plain, verbose, check)
end

function BasicGibbsJob(
  model::GenericModel,
  dpjob::Vector,
  range::BasicMCRange,
  v0::Vector;
  dpindex::IntegerVector=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Vector=[Dict(:destination=>:nstate, :monitor=>[:value]) for i in 1:length(dpindex)],
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
)
  vstate = default_state(model.vertices, v0, outopts, dpindex)
  BasicGibbsJob(model, dpindex, dpjob, range, vstate, outopts, plain, verbose, check)
end

function BasicGibbsJob(
  model::GenericModel,
  dpjob::Dict,
  range::BasicMCRange,
  v0::Dict;
  dpindex::IntegerVector=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Dict=Dict([(k, Dict(:destination=>:nstate, :monitor=>[:value])) for k in keys(model.vertices[dpindex])]),
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
)
  vstate = Dict{Symbol, VariableState}()
  for (k, v) in v0
    vstate[k] = default_state(model[k], v, get!(outopts, k, Dict()))
  end

  BasicGibbsJob(
    model,
    dpjob,
    range,
    vstate,
    dpindex=dpindex,
    outopts=outopts,
    plain=plain,
    verbose=verbose,
    check=check
  )
end

codegen(f::Symbol, job::BasicGibbsJob) = codegen(Val{f}, job)

function codegen(::Type{Val{:close}}, job::BasicGibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.output[j], VariableIOStream)
      push!(body, :(close(_job.output[$j])))
    end
  end

  @gensym _close

  quote
    function $_close(_job::BasicGibbsJob)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:resetplain}}, job::BasicGibbsJob)
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
    function $_resetplain(_job::BasicGibbsJob)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:resettask}}, job::BasicGibbsJob)
  @gensym _reset
  Expr(:function, Expr(:call, _reset, :(_job::BasicGibbsJob)), Expr(:block, :(_job.task.storage[:reset](_job))))
end

function codegen(::Type{Val{:save}}, job::BasicGibbsJob)
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
    function $_save(_job::BasicGibbsJob, _i::Integer)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:iterate}}, job::BasicGibbsJob)
  body = []

  for j in 1:job.ndp
    if isa(job.dependent[j], Parameter)
      if isa(job.dpjob[j], BasicMCJob)
        push!(body, :(run(_job.dpjob[$j])))
        push!(body, :(_job.dpstate[$j] = _job.dpjob[$j].pstate))
      else
        push!(body, :(setpdf!(_job.dependent[$j], _job.dpstate[$j])))
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
    function $_iterate(_job::BasicGibbsJob)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:run}}, job::BasicGibbsJob)
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
    function $_run(_job::BasicGibbsJob)
      $(body...)
    end
  end

  result
end

function checkin(job::BasicGibbsJob)
  dpindex = find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), job.model.vertices)
  ndp = length(dpindex)

  if ndp <= 0
    error("The model has neither parameters nor transformations, but at least one of them is required in a BasicGibbsJob")
  end

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

num_dp(job::BasicGibbsJob) = job.ndp
num_dptransforms(job::BasicGibbsJob) = count(v -> isa(v, Transformation), job.dependent)
num_randdp(job::BasicGibbsJob) = count(v -> isa(v, Parameter), job.dependent)
num_randdp_viamcmc(job::BasicGibbsJob) = count(j -> j != nothing, job.dpjob)
num_radndp_viadistribution(job::BasicGibbsJob) = num_dp(job)-num_dptransforms(job)-num_randdp_viamcmc(job)

dpkeys(job::BasicGibbsJob) = Symbol[dp.key for dp in job.dependent]

output(job::BasicGibbsJob) = job.output

Dict(job::BasicGibbsJob, field::Symbol=:output) = Dict(zip(dpkeys(job), getfield(job, field)))

function Base.show(io::IO, job::BasicGibbsJob)
  ndptransforms = num_dptransforms(job)
  ndpviamcmc = num_randdp_viamcmc(job)
  ndpviadistribution = job.ndp-ndptransforms-ndpviamcmc

  isplain = job.plain ? "job flow not controlled by tasks" : "job flow controlled by tasks"

  indentation = "  "

  println(io, "BasicGibbsJob:")
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
end

Base.writemime(io::IO, ::MIME"text/plain", job::BasicGibbsJob) = show(io, job)

function job2dot(stream::IOStream, job::BasicGibbsJob)
  graphkeyword, edgesign = is_directed(job.model) ? ("digraph", "->") : ("graph", "--")
  dotindentation, dotspacing = "  ", " "

  write(stream, "$graphkeyword BasicGibbsJob {\n")

  invdpindex = Dict{Symbol, Integer}(zip(keys(job.model.vertices[job.dpindex]), 1:job.ndp))

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

function job2dot(filename::AbstractString, job::BasicGibbsJob, mode::AbstractString="w")
  stream = open(filename, mode)
  job2dot(stream, job)
  close(stream)
end
