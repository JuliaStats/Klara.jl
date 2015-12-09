### MCJob

abstract MCJob

# initialize_output() needs to be defined for custom variable state or NState input arguments
# Thus multiple dispatch allows to extend the code base to accommodate new variable states or NStates

function initialize_output(state::BasicContUnvParameterState, n::Int, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicContUnvParameterNState(n, outopts[:monitor], outopts[:diagnostics], eltype(state))
  elseif outopts[:destination] == :iostream
    output = BasicContParamIOStream(
      (),
      n,
      outopts[:monitor],
      diagnostickeys=outopts[:diagnostics],
      mode="w",
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix]
    )
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

function initialize_output(state::BasicContMuvParameterState, n::Int, outopts::Dict)
  output::Union{VariableNState, VariableIOStream, Void}

  if outopts[:destination] == :nstate
    output = BasicContMuvParameterNState(state.size, n, outopts[:monitor], outopts[:diagnostics], eltype(state))
  elseif outopts[:destination] == :iostream
    output = BasicContParamIOStream(
      (),
      n,
      outopts[:monitor],
      diagnostickeys=outopts[:diagnostics],
      mode="w",
      filepath=outopts[:filepath],
      filesuffix=outopts[:filesuffix]
    )
    output.mark()
  elseif outopts[:destination] == :none
    output = nothing
  else
    error(":destination must be set to :nstate or :iostream or :none, got $(outopts[:destination])")
  end

  output
end

Base.run{J<:MCJob}(job::Vector{J}) = map(run, job)
