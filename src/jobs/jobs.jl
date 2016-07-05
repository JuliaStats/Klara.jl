### MCJob

abstract MCJob

abstract GibbsJob <: MCJob

# Set defaults for possibly unspecified output options

function augment_variable_outopts!(outopts::Dict)
  destination = get!(outopts, :destination, :nstate)

  if destination != :none
    if !haskey(outopts, :monitor)
      outopts[:monitor] = [:value]
    end

    if destination == :iostream
      if !haskey(outopts, :filepath)
        outopts[:filepath] = ""
      end

      if !haskey(outopts, :filesuffix)
        outopts[:filesuffix] = "csv"
      end

      if !haskey(outopts, :flush)
        outopts[:flush] = false
      end
    end
  end
end

augment_variable_outopts!{K, V}(outopts::Vector{Dict{K, V}}) = map(augment_variable_outopts!, outopts)

function augment_parameter_outopts!(outopts::Dict)
  augment_variable_outopts!(outopts)

  if outopts[:destination] != :none
    if !haskey(outopts, :diagnostics)
      outopts[:diagnostics] = Symbol[]
    end
  end
end

augment_parameter_outopts!{K, V}(outopts::Vector{Dict{K, V}}) = map(augment_parameter_outopts!, outopts)

# initialize_output() needs to be defined for custom variable state or NState input arguments
# Thus multiple dispatch allows to extend the code base to accommodate new variable states or NStates

function initialize_output(state::BasicUnvVariableState, n::Integer, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicUnvVariableNState(n, eltype(state))
  elseif outopts[:destination] == :iostream
    output = BasicVariableIOStream(
      (),
      n,
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix],
      mode="w"
    )
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_output(state::BasicMuvVariableState, n::Integer, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicMuvVariableNState(state.size, n, eltype(state))
  elseif outopts[:destination] == :iostream
    output = BasicVariableIOStream(
      (state.size,),
      n,
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix],
      mode="w"
    )
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_output(state::BasicMavVariableState, n::Integer, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicMavVariableNState(state.size, n, eltype(state))
  elseif outopts[:destination] == :iostream
    output = BasicVariableIOStream(
      state.size,
      n,
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix],
      mode="w"
    )
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_output(state::BasicDiscUnvParameterState, n::Integer, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicDiscUnvParameterNState(n, outopts[:monitor], outopts[:diagnostics], eltype(state)...)
  elseif outopts[:destination] == :iostream
    output = BasicDiscParamIOStream(
      (),
      n,
      outopts[:monitor],
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix],
      diagnostickeys=outopts[:diagnostics],
      mode="w"
    )
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_output(state::BasicDiscMuvParameterState, n::Integer, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicDiscMuvParameterNState(state.size, n, outopts[:monitor], outopts[:diagnostics], eltype(state)...)
  elseif outopts[:destination] == :iostream
    output = BasicDiscParamIOStream(
      (state.size,),
      n,
      outopts[:monitor],
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix],
      diagnostickeys=outopts[:diagnostics],
      mode="w"
    )
    mark(output)
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_output(state::BasicContUnvParameterState, n::Integer, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicContUnvParameterNState(n, outopts[:monitor], outopts[:diagnostics], eltype(state))
  elseif outopts[:destination] == :iostream
    output = BasicContParamIOStream(
      (),
      n,
      outopts[:monitor],
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix],
      diagnostickeys=outopts[:diagnostics],
      mode="w"
    )
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_output(state::BasicContMuvParameterState, n::Integer, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicContMuvParameterNState(state.size, n, outopts[:monitor], outopts[:diagnostics], eltype(state))
  elseif outopts[:destination] == :iostream
    output = BasicContParamIOStream(
      (state.size,),
      n,
      outopts[:monitor],
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix],
      diagnostickeys=outopts[:diagnostics],
      mode="w"
    )
    mark(output)
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_task!(job::MCJob)
  # Hook inside task to allow remote resetting
  task_local_storage(:reset, job.resetplain!)

  while true
    iterate(job)
  end
end

Base.close(job::MCJob) = job.close(job)

Base.reset(job::MCJob) = job.reset!(job)
Base.reset(job::MCJob, x::Real) = job.reset!(job, x)
Base.reset{N<:Real}(job::MCJob, x::Vector{N}) = job.reset!(job, x)

save(job::MCJob, i::Integer) = job.save!(job, i)

iterate(job::MCJob) = job.iterate!(job)

Base.run(job::MCJob) = job.run!(job)
Base.run{J<:MCJob}(job::Vector{J}) = map(run, job)
