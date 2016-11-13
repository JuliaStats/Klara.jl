### Abstract parameter IOStreams

abstract ParameterIOStream{S<:ValueSupport} <: VariableIOStream

codegen(f::Symbol, iostream::ParameterIOStream, fnames::Vector{Symbol}) = codegen(Val{f}, iostream, fnames)

open{S<:AbstractString}(iostream::ParameterIOStream, names::Vector{S}, mode::AbstractString="w") =
  iostream.open(iostream, names, mode)

close(iostream::ParameterIOStream) = iostream.close(iostream)

mark(iostream::ParameterIOStream) = iostream.mark(iostream)

reset(iostream::ParameterIOStream) = iostream.reset(iostream)

flush(iostream::ParameterIOStream) = iostream.flush(iostream)

write{S<:ValueSupport, F<:VariateForm}(iostream::ParameterIOStream{S}, state::ParameterState{S, F}) =
  iostream.write(iostream, state)
