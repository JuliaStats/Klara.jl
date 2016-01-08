### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

type BasicMCJob{S<:VariableState} <: MCJob
  model::GenericModel # Model of single parameter
  pindex::Int # Index of single parameter in model.vertices
  parameter::Parameter # Points to model.vertices[pindex] for faster access
  sampler::MCSampler
  tuner::MCTuner
  range::BasicMCRange
  vstate::Vector{S} # Vector of variable states ordered according to variables in model.vertices
  pstate::ParameterState # Points to vstate[pindex] for faster access
  sstate::MCSamplerState # Internal state of MCSampler
  outopts::Dict # Options related to output
  output::Union{VariableNState, VariableIOStream, Void} # Output of model's single parameter
  count::Int # Current number of post-burnin iterations
  resetpstate::Bool # If resetpstate=true then pstate is reset by reset(job), else pstate is not modified by reset(job)
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  task::Union{Task, Void}
  resetplain!::Function
  reset!::Function
  save!::Union{Function, Void}
  iterate!::Function
  run!::Function

  function BasicMCJob(
    model::GenericModel,
    pindex::Int,
    sampler::MCSampler,
    tuner::MCTuner,
    range::BasicMCRange,
    vstate::Vector{S},
    outopts::Dict,
    resetpstate::Bool,
    plain::Bool,
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

    instance.sstate = sampler_state(sampler, tuner, instance.pstate)

    instance.outopts = isa(outopts, Dict{Symbol, Any}) ? outopts : convert(Dict{Symbol, Any}, outopts)
    augment_parameter_outopts!(instance.outopts)

    instance.output = initialize_output(instance.pstate, range.npoststeps, instance.outopts)

    instance.count = 0

    instance.resetplain! = eval(codegen_resetplain_basicmcjob(instance))

    instance.save! = (instance.output == nothing) ? nothing : eval(codegen_save_basicmcjob(instance))

    instance.iterate! = eval(codegen_iterate_basicmcjob(instance))

    if plain
      instance.task = nothing
      instance.reset! = instance.resetplain!
    else
      instance.task = Task(() -> initialize_task!(instance))
      instance.reset! = eval(codegen_reset_task_basicmcjob(instance))
    end

    instance.run! = eval(codegen_run_basicmcjob(instance))

    instance
  end
end

BasicMCJob{S<:VariableState}(
  model::GenericModel,
  pindex::Int,
  sampler::MCSampler,
  tuner::MCTuner,
  range::BasicMCRange,
  vstate::Vector{S},
  outopts::Dict,
  resetpstate::Bool,
  plain::Bool,
  check::Bool
) =
  BasicMCJob{S}(model, pindex, sampler, tuner, range, vstate, outopts, resetpstate, plain, check)

BasicMCJob{S<:VariableState}(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Vector{S};
  pindex::Int=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
  check::Bool=false
) =
  BasicMCJob(model, pindex, sampler, tuner, range, v0, outopts, resetpstate, plain, check)

function BasicMCJob{S<:VariableState}(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Dict{Symbol, S};
  pindex::Int=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
  check::Bool=false
)
  vstate = Array(S, length(v0))
  for (k, v) in v0
    vstate[model.ofkey[k]] = v
  end

  BasicMCJob(model, pindex, sampler, tuner, range, vstate, outopts, resetpstate, plain, check)
end

function BasicMCJob(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Vector;
  pindex::Int=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
  check::Bool=false
)
  nv0 = length(v0)
  vstate = Array(VariableState, nv0)
  for i in 1:nv0
    if isa(v0[i], VariableState)
      vstate[i] = v0[i]
    elseif isa(v0[i], Number) ||
      (isa(v0[i], Vector) && issubtype(eltype(v0[i]), Number)) ||
      (isa(v0[i], Matrix) && issubtype(eltype(v0[i]), Number))
      if isa(model.vertices[pindex], Parameter)
        vstate[i] = default_state(model.vertices[i], v0[i], outopts)
      else
        vstate[i] = default_state(model.vertices[i], v0[i])
      end
    else
      error("Variable state or state value of type $(typeof(v0[i])) not valid")
    end
  end

  BasicMCJob(model, pindex, sampler, tuner, range, vstate, outopts, resetpstate, plain, check)
end

function BasicMCJob(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Dict;
  pindex::Int=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  plain::Bool=true,
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
    check=check
  )
end

# It is likely that MCMC inference for parameters of ODEs will require a separate ODEBasicMCJob
# In that case the iterate!() function will take a second variable (transformation) as input argument

function codegen_resetplain_basicmcjob(job::BasicMCJob)
  fsignature::Vector{Union{Symbol, Expr}}
  body = []

  if job.resetpstate
    push!(body, :(reset!($(job).pstate, _x, $(job).parameter, $(job).sampler)))
  end

  push!(body, :(reset!($(job).sstate.tune, $(job).sampler, $(job).tuner)))

  if isa(job.output, VariableIOStream)
    push!(body, :($(job).output.reset()))
    push!(body, :($(job).output.mark()))
  end

  push!(body, :($(job).count = 0))

  @gensym resetplain_basicmcjob

  if job.resetpstate
    vform = variate_form(job.pstate)
    if vform == Univariate
      fsignature = Union{Symbol, Expr}[resetplain_basicmcjob, :(_x::Real)]
    elseif vform == Multivariate
      fsignature = Union{Symbol, Expr}[:($resetplain_basicmcjob{N<:Real}), :(_x::Vector{N})]
    else
      error("It is not possible to define plain reset for given job")
    end
  else
    fsignature = Union{Symbol, Expr}[resetplain_basicmcjob]
  end

  Expr(:function, Expr(:call, fsignature...), Expr(:block, body...))
end

function codegen_reset_task_basicmcjob(job::BasicMCJob)
  fsignature::Vector{Union{Symbol, Expr}}
  fbody::Expr

  @gensym reset_task_basicmcjob

  if job.resetpstate
    vform = variate_form(job.pstate)
    if vform == Univariate
      fsignature = Union{Symbol, Expr}[reset_task_basicmcjob, :(_x::Real)]
    elseif vform == Multivariate
      fsignature = Union{Symbol, Expr}[:($reset_task_basicmcjob{N<:Real}), :(_x::Vector{N})]
    else
      error("It is not possible to define plain reset for given job")
    end
    fbody = :($(job).task.storage[:reset](_x))
  else
    fsignature = Union{Symbol, Expr}[reset_task_basicmcjob]
    fbody = :($(job).task.storage[:reset]())
  end

  Expr(:function, Expr(:call, fsignature...), Expr(:block, fbody))
end

Base.reset(job::BasicMCJob) = job.reset!()
Base.reset(job::BasicMCJob, x::Real) = job.reset!(x)
Base.reset{N<:Real}(job::BasicMCJob, x::Vector{N}) = job.reset!(x)

function codegen_save_basicmcjob(job::BasicMCJob)
  body = []

  if isa(job.output, VariableNState)
    push!(body, :($(job).output.copy($(job).pstate, _i)))
  elseif isa(job.output, VariableIOStream)
    push!(body, :($(job).output.write($(job).pstate)))
    if job.outopts[:flush]
      push!(body, :($(job).output.flush()))
    end
  else
    error("To save output, :destination must be set to :nstate or :iostream, got $(job.outopts[:destination])")
  end

  @gensym save_basicmcjob

  quote
    function $save_basicmcjob(_i::Int)
      $(body...)
    end
  end
end

function codegen_run_basicmcjob(job::BasicMCJob)
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

  push!(body, Expr(:for, :(i = 1:$(job).range.nsteps), Expr(:block, forbody...)))

  if isa(job.output, VariableIOStream)
    push!(body, :($(job).output.close()))
  end

  @gensym run_basicmcjob

  result = quote
    function $run_basicmcjob()
      $(body...)
    end
  end

  result
end

Base.run(job::BasicMCJob) = job.run!()

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

  println(io, "BasicMCJob:")
  print(io, "  ")
  show(io, job.parameter)
  print(io, "\n  ")
  show(io, job.model)
  print(io, "\n  ")
  show(io, job.sampler)
  print(io, "\n  ")
  show(io, job.tuner)
  print(io, "\n  ")
  show(io, job.range)
  print(io, "\n  plain = $(job.plain) ($isplain)")
end

Base.writemime(io::IO, ::MIME"text/plain", job::BasicMCJob) = show(io, job)
