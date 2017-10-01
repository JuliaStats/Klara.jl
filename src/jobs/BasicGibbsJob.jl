### BasicGibbsJob

mutable struct BasicGibbsJob <: GibbsJob
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
  hasdpjob::Bool
  hasoutput::Bool
  hasiostream::Bool
  count::Integer # Current number of post-burnin iterations
  verbose::Bool

  function BasicGibbsJob(
    model::GenericModel,
    dpindex::IntegerVector,
    dpjob::Vector,
    range::BasicMCRange,
    vstate::VariableStateVector,
    outopts::Vector,
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

    instance.hasdpjob = any(j -> j != nothing, instance.dpjob)

    instance.outopts = isa(outopts, Vector{Dict{Symbol, Any}}) ? outopts : convert(Vector{Dict{Symbol, Any}}, outopts)

    instance.range = range
    instance.verbose = verbose

    instance.dependent = instance.model.vertices[instance.dpindex]
    for i in 1:instance.ndp
      instance.dependent[i].states = instance.vstate
    end

    instance.dpstate = instance.vstate[instance.dpindex]

    instance.output = Array{Union{VariableNState, VariableIOStream, Void}}(instance.ndp)
    for i in 1:instance.ndp
      if isa(instance.dependent[i], Parameter)
        augment_parameter_outopts!(instance.outopts[i])
      else
        augment_variable_outopts!(instance.outopts[i])
      end
      instance.output[i] = initialize_output(instance.dpstate[i], range.npoststeps, instance.outopts[i])
    end

    instance.hasoutput = any(o -> o != nothing, instance.output)

    instance.hasiostream = any(o -> o == VariableIOStream, instance.output)

    instance.count = 0

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
  verbose::Bool=false,
  check::Bool=false
) =
  BasicGibbsJob(model, dpindex, dpjob, range, v0, outopts, verbose, check)

function BasicGibbsJob(
  model::GenericModel,
  dpjob::Dict,
  range::BasicMCRange,
  v0::Dict{Symbol, S};
  dpindex::IntegerVector=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Dict=Dict([(k, Dict(:destination=>:nstate, :monitor=>[:value])) for k in keys(model.vertices[dpindex])]),
  verbose::Bool=false,
  check::Bool=false
) where S<:VariableState
  ndpindex = length(dpindex)

  jobs = Array{Union{BasicMCJob, Void}}(ndpindex)
  opts = Array{Dict}(ndpindex)

  for i in 1:ndpindex
    key = vertex_key(model.vertices[dpindex[i]])
    jobs[i] = haskey(dpjob, key) ? dpjob[key] : nothing
    opts[i] = haskey(outopts, key) ? outopts[key] : Dict(:destination=>:nstate, :monitor=>[:value])
  end

  vstate = Array{S}(length(v0))
  for (k, v) in v0
    vstate[model.ofkey[k]] = v
  end

  BasicGibbsJob(model, dpindex, jobs, range, vstate, opts, verbose, check)
end

function BasicGibbsJob(
  model::GenericModel,
  dpjob::Vector,
  range::BasicMCRange,
  v0::Vector;
  dpindex::IntegerVector=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Vector=[Dict(:destination=>:nstate, :monitor=>[:value]) for i in 1:length(dpindex)],
  verbose::Bool=false,
  check::Bool=false
)
  vstate = default_state(model.vertices, v0, outopts, dpindex)
  BasicGibbsJob(model, dpindex, dpjob, range, vstate, outopts, verbose, check)
end

function BasicGibbsJob(
  model::GenericModel,
  dpjob::Dict,
  range::BasicMCRange,
  v0::Dict;
  dpindex::IntegerVector=find(v::Variable -> isa(v, Parameter) || isa(v, Transformation), model.vertices),
  outopts::Dict=Dict([(k, Dict(:destination=>:nstate, :monitor=>[:value])) for k in keys(model.vertices[dpindex])]),
  verbose::Bool=false,
  check::Bool=false
)
  vstate = Dict{Symbol, VariableState}()
  for (k, v) in v0
    vstate[k] = default_state(model[k], v, get!(outopts, k, Dict()))
  end

  BasicGibbsJob(model, dpjob, range, vstate, dpindex=dpindex, outopts=outopts, verbose=verbose, check=check)
end

function close(job::BasicGibbsJob)
  for i in 1:job.ndp
    if isa(job.output[i], VariableIOStream)
      close(job.output[i])
    end
  end
end

function reset(job::BasicGibbsJob)
  for i in 1:job.ndp
    if isa(job.dpjob[i], BasicMCJob)
      if job.dpjob[i].resetpstate
        reset(job.dpjob[i], rand(job.dependent[i].prior))
      else
        reset(job.dpjob[i])
      end
    end
  end
end

function save(job::BasicGibbsJob, i::Integer)
  for j in 1:job.ndp
    if isa(job.output[j], VariableNState)
      copy!(job.output[j], job.dpstate[j], i)
    elseif isa(job.output[j], VariableIOStream)
      write(job.output[j], job.dpstate[j])
      if job.outopts[j][:flush]
        flush(job.outopts[j])
      end
    else
      error("To save output, :destination must be set to :nstate or :iostream, got $(job.outopts[j][:destination])")
    end
  end
end

function iterate!(job::BasicGibbsJob)
  for i in 1:job.ndp
    if isa(job.dependent[i], Parameter)
      if isa(job.dpjob[i], BasicMCJob)
        run(job.dpjob[i])
        job.dpstate[i] = job.dpjob[i].pstate
      else
        setpdf!(job.dependent[i], job.dpstate[i])
        job.dpstate[i].value = rand(job.dependent[i].pdf)
      end
    else
      job.dependent[i].transform!(job.dpstate[i])
    end
  end
end

function run(job::BasicGibbsJob)
  fmt_iter::Union{Function, Void} = job.verbose ? format_iteration(ndigits(job.range.nsteps)) : nothing

  if job.verbose
    println("Iteration ", fmt_iter(i), " of ", job.range.nsteps)
  end

  for i = 1:job.range.nsteps
    iterate!(job)

    if in(i, job.range.postrange)
      job.count += 1

      if job.hasoutput
        save(job, job.count)
      end
    end

    if job.hasdpjob
      reset(job)
    end
  end

  if isa(job.output, VariableIOStream)
    close(job.output)
  end

  if job.hasiostream
    close(job)
  end
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

function show(io::IO, job::BasicGibbsJob)
  ndptransforms = num_dptransforms(job)
  ndpviamcmc = num_randdp_viamcmc(job)
  ndpviadistribution = job.ndp-ndptransforms-ndpviamcmc

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
end

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
