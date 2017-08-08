### Abstract variable IOStreams

abstract type VariableIOStream end

### BasicVariableIOStream

mutable struct BasicVariableIOStream <: VariableIOStream
  stream::IOStream
  size::Tuple
  n::Integer
end

BasicVariableIOStream(size::Tuple, n::Integer, filename::AbstractString, mode::AbstractString="w") =
  BasicVariableIOStream(open(filename, mode), size, n)

BasicVariableIOStream(
  size::Tuple,
  n::Integer;
  filepath::AbstractString="",
  filesuffix::AbstractString="csv",
  mode::AbstractString="w"
) =
  BasicVariableIOStream(size, n, joinpath(filepath, "value."*filesuffix), mode)

close(iostream::BasicVariableIOStream) = close(iostream.stream)

write(iostream::BasicVariableIOStream, state::VariableState) = write(iostream.stream, join(state.value, ','), "\n")

write(iostream::BasicVariableIOStream, nstate::BasicUnvVariableNState) = writedlm(iostream.stream, nstate.value)

write(iostream::BasicVariableIOStream, nstate::BasicMuvVariableNState) = writedlm(iostream.stream, nstate.value', ',')

function write(iostream::BasicVariableIOStream, nstate::BasicMavVariableNState)
  statelen = prod(nstate.size)
  for i in 1:nstate.n
    write(iostream.stream, join(nstate.value[1+(i-1)*statelen:i*statelen], ','), "\n")
  end
end

read!(iostream::BasicVariableIOStream, nstate::BasicUnvVariableNState{N}) where {N<:Number} =
  nstate.value = vec(readdlm(iostream.stream, ',', N))

read!(iostream::BasicVariableIOStream, nstate::BasicMuvVariableNState{N}) where {N<:Number} =
  nstate.value = readdlm(iostream.stream, ',', N)'

function read!(iostream::BasicVariableIOStream, nstate::BasicMavVariableNState{N}) where N<:Number
  statelen = prod(iostream.size)
  line = 1
  while !eof(iostream.stream)
    nstate.value[1+(line-1)*statelen:line*statelen] = [parse(N, c) for c in split(chomp(readline(iostream.stream)), ',')]
    line += 1
  end
end

function read(iostream::BasicVariableIOStream, T::Type{N}) where N<:Number
  local nstate::VariableNState
  l = length(iostream.size)

  if l == 0
    nstate = BasicUnvVariableNState(iostream.n, T)
  elseif l == 1
    nstate = BasicMuvVariableNState(iostream.size[1], iostream.n, T)
  elseif l == 2
    nstate = BasicMavVariableNState(iostream.size, iostream.n, T)
  else
    error("BasicVariableIOStream.size must be a tuple of length 0 or 1 or 2, got $(iostream.size) length")
  end

  read!(iostream, nstate)

  nstate
end

show(io::IO, iostream::BasicVariableIOStream) =
  print(io, "BasicVariableIOStream: state size = $(iostream.size), number of states = $(iostream.n)")
