### Abstract parameter IOStreams

abstract type ParameterIOStream{S<:ValueSupport} <: VariableIOStream end

codegen(f::Symbol, iostream::ParameterIOStream, fnames::Vector{Symbol}) = codegen(Val{f}, iostream, fnames)

open(iostream::ParameterIOStream, names::Vector{S}, mode::AbstractString="w") where {S<:AbstractString} =
  iostream.open(iostream, names, mode)

close(iostream::ParameterIOStream) = iostream.close(iostream)

mark(iostream::ParameterIOStream) = iostream.mark(iostream)

reset(iostream::ParameterIOStream) = iostream.reset(iostream)

flush(iostream::ParameterIOStream) = iostream.flush(iostream)

write(iostream::ParameterIOStream{S}, state::ParameterState{S, F}) where {S<:ValueSupport, F<:VariateForm} =
  iostream.write(iostream, state)
