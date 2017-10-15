### BasicMCJob

# BasicMCJob is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) job

mutable struct BasicMCJob <: MCJob
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
  fmt_iter::Union{Function, Void}
  fmt_perc::Union{Function, Void}
  resetpstate::Bool # If resetpstate=true then pstate is reset by reset(job), else pstate is not modified by reset(job)
  verbose::Bool

  function BasicMCJob(
    model::GenericModel,
    pindex::Integer,
    sampler::MCSampler,
    tuner::MCTuner,
    range::BasicMCRange,
    vstate::VariableStateVector,
    outopts::Dict,
    resetpstate::Bool,
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

    instance.outopts = isa(outopts, Dict{Symbol, Any}) ? outopts : convert(Dict{Symbol, Any}, outopts)
    augment_parameter_outopts!(instance.outopts)

    instance.parameter = instance.model.vertices[instance.pindex]
    instance.parameter.states = instance.vstate

    instance.sampler = sampler
    instance.tuner = tuner
    instance.range = range
    instance.verbose = verbose

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
    if value_support(instance.parameter) == Continuous &&
      instance.parameter.diffopts != nothing &&
      instance.parameter.diffopts.mode == :reverse
      set_tapes!(instance.parameter, instance.pstate)
    end
    initialize!(instance.pstate, instance.parameter, instance.sampler, instance.outopts)
    if value_support(instance.parameter) == Continuous
      instance.parameter.state = instance.pstate
    end

    instance.sstate = sampler_state(
      instance.parameter, instance.sampler, tuner, instance.pstate, instance.vstate, instance.outopts[:diagnostics]
    )

    if isa(instance.sampler, MuvAMWG)
      initialize_diagnosticvalues!(instance.pstate, instance.sstate)
    end

    instance.output = initialize_output(instance.pstate, range.npoststeps, instance.outopts)

    instance.count = 0

    instance.fmt_iter =
      if instance.tuner.verbose
        if isa(sampler, NUTS) && isa(instance.tuner, DualAveragingMCTuner)
          format_iteration(ndigits(instance.tuner.nadapt))
        else
          format_iteration(ndigits(range.burnin))
        end
      else
        nothing
      end

    instance.fmt_perc = instance.tuner.verbose ? format_percentage() : nothing

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
  verbose::Bool=false,
  check::Bool=false
) =
  BasicMCJob(model, pindex, sampler, tuner, range, v0, outopts, resetpstate, verbose, check)

function BasicMCJob(
  model::GenericModel,
  sampler::MCSampler,
  range::BasicMCRange,
  v0::Dict{Symbol, S};
  pindex::Integer=findfirst(v::Variable -> isa(v, Parameter), model.vertices),
  tuner::MCTuner=VanillaMCTuner(),
  outopts::Dict=Dict(:destination=>:nstate, :monitor=>[:value], :diagnostics=>Symbol[]),
  resetpstate::Bool=true,
  verbose::Bool=false,
  check::Bool=false
) where S<:VariableState
  vstate = Array{S}(length(v0))
  for (k, v) in v0
    vstate[model.ofkey[k]] = v
  end
  BasicMCJob(model, pindex, sampler, tuner, range, vstate, outopts, resetpstate, verbose, check)
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
  verbose::Bool=false,
  check::Bool=false
)
  vstate = default_state(model.vertices, v0, [outopts], [pindex])
  BasicMCJob(model, pindex, sampler, tuner, range, vstate, outopts, resetpstate, verbose, check)
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
  verbose::Bool=false,
  check::Bool=false
)
  vstate = Array{Any}(length(v0))
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
    verbose=verbose,
    check=check
  )
end

function reset(job::BasicMCJob)
  reset!(job.sstate, job.pstate, job.parameter, job.sampler, job.tuner)

  if isa(job.output, VariableIOStream)
    reset(job.output)
    mark(job.output)
  end

  job.count = 0
end

function reset(job::BasicMCJob, x)
  reset!(job.pstate, x, job.parameter, job.sampler)
  reset(job)
end

function save(job::BasicMCJob)
  write(job.output, job.pstate)
  if job.outopts[:flush]
    flush(job.output)
  end
end

save(job::BasicMCJob, i::Integer) = copy!(job.output, job.pstate, i)

function run(job::BasicMCJob)
  fmt_iter::Union{Function, Void} = job.verbose ? format_iteration(ndigits(job.range.nsteps)) : nothing

  if isa(job.output, VariableIOStream)
    mark(job.output)
  end

  for i = 1:job.range.nsteps
    if job.verbose
      println("Iteration ", fmt_iter(i), " of ", job.range.nsteps)
    end

    iterate!(job, typeof(job.sampler), variate_form(job.parameter))

    if in(i, job.range.postrange)
      job.count += 1

      if job.output != nothing
        if isa(job.output, VariableNState)
          save(job, job.count)
        elseif isa(job.output, VariableIOStream)
          save(job)
        else
          error("To save output, :destination must be set to :nstate or :iostream, got $(job.outopts[:destination])")
        end
      end
    end
  end

  if isa(job.output, VariableIOStream)
    close(job.output)
  end
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

function show(io::IO, job::BasicMCJob)
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
end
