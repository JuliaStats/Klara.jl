### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

type BasicMCJob <: MCJob
  model::GenericModel # Model of single parameter
  pindex::Integer # Index of single parameter in model.vertices
  parameter::Parameter # Points to model.vertices[pindex] for faster access
  sampler::MCSampler
  tuner::MCTuner
  range::BasicMCRange
  vstate::VariableStateVector # Vector of variable states ordered according to variables in model.vertices
  pstate::ParameterState # Points to vstate[pindex] for faster access
  sstate::MCSamplerState # Internal state of MCSampler
  outopts::Dict # Options related to output
  output::Union{VariableNState, VariableIOStream, Void} # Output of model's single parameter
  count::Integer # Current number of post-burnin iterations
  resetpstate::Bool # If resetpstate=true then pstate is reset by reset(job), else pstate is not modified by reset(job)
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  task::Union{Task, Void}
  verbose::Bool
  resetplain!::Function
  reset!::Function
  save!::Union{Function, Void}
  iterate!::Function
  run!::Function

  function BasicMCJob(
    model::GenericModel,
    pindex::Integer,
    sampler::MCSampler,
    tuner::MCTuner,
    range::BasicMCRange,
    vstate::VariableStateVector,
    outopts::Dict,
    resetpstate::Bool,
    plain::Bool,
    verbose::Bool,
    check::Bool
  )
    instance = new()

    instance.model = model
    instance.pindex = pindex
    instance.vstate = vstate
    instance.resetpstate = resetpstate

    if check
      checkin(instance)
    end

    instance.sampler = sampler
    instance.tuner = tuner
    instance.range = range
    instance.plain = plain
    instance.verbose = verbose

    instance.parameter = instance.model.vertices[instance.pindex]
    instance.parameter.states = instance.vstate

    instance.pstate = instance.vstate[instance.pindex]
    if any(isnan, instance.pstate.value)
      if instance.parameter.pdf != nothing
        instance.pstate.value = rand(instance.parameter.pdf)
      elseif instance.parameter.prior != nothing
        instance.pstate.value = rand(instance.parameter.prior)
      else
        error("Not possible to initialize pstate with missing pstate.value and without parameter.pdf or parameter.prior")
      end
    end
    initialize!(instance.pstate, instance.parameter, sampler)

    instance.sstate = sampler_state(instance.parameter, sampler, tuner, instance.pstate, instance.vstate)

    instance.outopts = isa(outopts, Dict{Symbol, Any}) ? outopts : convert(Dict{Symbol, Any}, outopts)
    augment_parameter_outopts!(instance.outopts)

    instance.output = initialize_output(instance.pstate, range.npoststeps, instance.outopts)

    instance.count = 0

    instance.resetplain! = eval(codegen(:resetplain, instance))

    instance.save! = (instance.output == nothing) ? nothing : eval(codegen(:save, instance))

    instance.iterate! = eval(codegen(:iterate, instance))

    if plain
      instance.task = nothing
      instance.reset! = instance.resetplain!
    else
      instance.task = Task(() -> initialize_task!(instance))
      instance.reset! = eval(codegen(:resettask, instance))
    end

    instance.run! = eval(codegen(:run, instance))

    instance
  end
end

BasicMCJob(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::VariableStateVector;
  pindex::Integer=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
) =
  BasicMCJob(model, pindex, sampler, tuner, range, v0, outopts, resetpstate, plain, verbose, check)

function BasicMCJob{S<:VariableState}(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Dict{Symbol, S};
  pindex::Integer=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
)
  vstate = Array(S, length(v0))
  for (k, v) in v0
    vstate[model.ofkey[k]] = v
  end
  BasicMCJob(model, pindex, sampler, tuner, range, vstate, outopts, resetpstate, plain, verbose, check)
end

function BasicMCJob(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Vector;
  pindex::Integer=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
)
  vstate = default_state(model.vertices, v0, [outopts], [pindex])
  BasicMCJob(model, pindex, sampler, tuner, range, vstate, outopts, resetpstate, plain, verbose, check)
end

function BasicMCJob(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Dict;
  pindex::Integer=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
  verbose::Bool=false,
  check::Bool=false
)
  vstate = Array(Any, length(v0))
  for (k, v) in v0
    vstate[model.ofkey[k]] = v
  end

  BasicMCJob(
    model,
    sampler,
    range,
    vstate,
    pindex=pindex,
    tuner=tuner,
    outopts=outopts,
    resetpstate=resetpstate,
    plain=plain,
    verbose=verbose,
    check=check
  )
end

# It is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob

codegen(f::Symbol, job::BasicMCJob) = codegen(Val{f}, job)

function codegen(::Type{Val{:resetplain}}, job::BasicMCJob)
  fsignature::Vector{Union{Symbol, Expr}}
  body = []

  if job.resetpstate
    push!(body, :(reset!(_job.pstate, _x, _job.parameter, _job.sampler)))
  end

  push!(body, :(reset!(_job.sstate, _job.pstate, _job.parameter, _job.sampler, _job.tuner)))

  if isa(job.output, VariableIOStream)
    push!(body, :(reset(_job.output)))
    push!(body, :(mark(_job.output)))
  end

  push!(body, :(_job.count = 0))

  @gensym _resetplain

  if job.resetpstate
    vform = variate_form(job.pstate)
    if vform == Univariate
      fsignature = Union{Symbol, Expr}[_resetplain, :(_job::BasicMCJob), :(_x::Real)]
    elseif vform == Multivariate
      fsignature = Union{Symbol, Expr}[:($_resetplain{N<:Real}), :(_job::BasicMCJob), :(_x::Vector{N})]
    else
      error("It is not possible to define plain reset for given job")
    end
  else
    fsignature = Union{Symbol, Expr}[_resetplain, :(_job::BasicMCJob)]
  end

  Expr(:function, Expr(:call, fsignature...), Expr(:block, body...))
end

function codegen(::Type{Val{:resettask}}, job::BasicMCJob)
  fsignature::Vector{Union{Symbol, Expr}}
  fbody::Expr

  @gensym _resettask

  if job.resetpstate
    vform = variate_form(job.pstate)
    if vform == Univariate
      fsignature = Union{Symbol, Expr}[_resettask, :(_job::BasicMCJob), :(_x::Real)]
    elseif vform == Multivariate
      fsignature = Union{Symbol, Expr}[:($_resettask{N<:Real}), :(_job::BasicMCJob), :(_x::Vector{N})]
    else
      error("It is not possible to define plain reset for given job")
    end
    fbody = :(_job.task.storage[:reset](_job, _x))
  else
    fsignature = Union{Symbol, Expr}[_resettask, :(_job::BasicMCJob)]
    fbody = :(_job.task.storage[:reset](_job))
  end

  Expr(:function, Expr(:call, fsignature...), Expr(:block, fbody))
end

function codegen(::Type{Val{:save}}, job::BasicMCJob)
  body = []

  if isa(job.output, VariableNState)
    push!(body, :(copy!(_job.output, _job.pstate, _i)))
  elseif isa(job.output, VariableIOStream)
    push!(body, :(write(_job.output, _job.pstate)))
    if job.outopts[:flush]
      push!(body, :(flush(_job.output)))
    end
  else
    error("To save output, :destination must be set to :nstate or :iostream, got $(job.outopts[:destination])")
  end

  @gensym _save

  quote
    function $_save(_job::BasicMCJob, _i::Integer)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:run}}, job::BasicMCJob)
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

  push!(body, Expr(:for, :(i = 1:_job.range.nsteps), Expr(:block, forbody...)))

  if isa(job.output, VariableIOStream)
    push!(body, :(close(_job.output)))
  end

  @gensym _run

  result = quote
    function $_run(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end

function checkin(job::BasicMCJob)
  pindex = find(v::Variable -> isa(v, Parameter), job.model.vertices)
  np = length(pindex)

  if np == 0
    error("The model does not have a parameter, but a BasicMCJob requires a parameter")
  elseif np == 1
    if pindex[1] != job.pindex
      error("Parameter located in model.vertices[$(pindex[1])], but pindex = $(job.pindex)")
    end
  elseif np >= 2
    warn("The model has $np parameters, but a BasicMCJob requires exactly one parameter")
    if in(job.pindex, pindex)
      error("Indices of parameters in model.vertices do not contain pindex")
    end
  end

  nv = num_vertices(job.model)
  nvstate = length(job.vstate)

  if nv != nvstate
    warn("Number of variables ($nv) not equal to number of variable states ($nvstate)")
  end

  pstate = job.vstate[job.pindex]

  if !isa(pstate, ParameterState)
    error("The parameter's state must be saved in a ParameterState subtype, got $(typeof(pstate)) state type")
  else
    check_support(job.model.vertices[job.pindex], pstate)
  end
end

output(job::BasicMCJob) = job.output

function Base.show(io::IO, job::BasicMCJob)
  isplain = job.plain ? "job flow not controlled by tasks" : "job flow controlled by tasks"

  indentation = "  "

  println(io, "BasicMCJob:")
  print(io, indentation)
  show(io, job.parameter)
  print(io, "\n"*indentation)
  show(io, job.model)
  print(io, "\n"*indentation)
  show(io, job.sampler)
  print(io, "\n"*indentation)
  show(io, job.tuner)
  print(io, "\n"*indentation)
  show(io, job.range)
  print(io, "\n"*indentation*"plain = $(job.plain) ($isplain)")
end

Base.writemime(io::IO, ::MIME"text/plain", job::BasicMCJob) = show(io, job)
