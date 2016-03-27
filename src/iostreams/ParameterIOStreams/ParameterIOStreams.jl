### Abstract parameter IOStreams

abstract ParameterIOStream{S<:ValueSupport} <: VariableIOStream

codegen(f::Symbol, iostream::ParameterIOStream, fnames::Vector{Symbol}) = codegen(Val{f}, iostream, fnames)

Base.open{S<:AbstractString}(iostream::ParameterIOStream, names::Vector{S}, mode::AbstractString="w") =
  iostream.open(iostream, names, mode)

close(iostream::ParameterIOStream) = iostream.close(iostream)

Base.mark(iostream::ParameterIOStream) = iostream.mark(iostream)

Base.reset(iostream::ParameterIOStream) = iostream.reset(iostream)

Base.flush(iostream::ParameterIOStream) = iostream.flush(iostream)

Base.write{S<:ValueSupport, F<:VariateForm}(iostream::ParameterIOStream{S}, state::ParameterState{S, F}) =
  iostream.write(iostream, state)

Base.writemime(io::IO, ::MIME"text/plain", iostream::ParameterIOStream) = show(io, iostream)
