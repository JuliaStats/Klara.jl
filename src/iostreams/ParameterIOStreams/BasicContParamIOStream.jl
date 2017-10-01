### BasicContParamIOStream

mutable struct BasicContParamIOStream <: ParameterIOStream{Continuous}
  value::Union{IOStream, Void}
  loglikelihood::Union{IOStream, Void}
  logprior::Union{IOStream, Void}
  logtarget::Union{IOStream, Void}
  gradloglikelihood::Union{IOStream, Void}
  gradlogprior::Union{IOStream, Void}
  gradlogtarget::Union{IOStream, Void}
  tensorloglikelihood::Union{IOStream, Void}
  tensorlogprior::Union{IOStream, Void}
  tensorlogtarget::Union{IOStream, Void}
  dtensorloglikelihood::Union{IOStream, Void}
  dtensorlogprior::Union{IOStream, Void}
  dtensorlogtarget::Union{IOStream, Void}
  diagnosticvalues::Union{IOStream, Void}
  names::Vector{AbstractString}
  size::Tuple
  n::Integer
  diagnostickeys::Vector{Symbol}

  function BasicContParamIOStream(
    size::Tuple,
    n::Integer,
    streams::Vector{Union{IOStream, Void}},
    diagnostickeys::Vector{Symbol}=Symbol[],
    filenames::Vector{AbstractString}=[(streams[i] == nothing) ? "" : streams[i].name[7:end-1] for i in 1:14]
  )
    instance = new()

    fnames = fieldnames(BasicContParamIOStream)
    for i in 1:14
      setfield!(instance, fnames[i], streams[i])
    end

    instance.names = filenames

    instance.size = size
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance
  end
end

function BasicContParamIOStream(
  size::Tuple,
  n::Integer,
  filenames::Vector{AbstractString},
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(BasicContParamIOStream)
  BasicContParamIOStream(
    size,
    n,
    Union{IOStream, Void}[isempty(filenames[i]) ? nothing : open(filenames[i], mode) for i in 1:14],
    diagnostickeys,
    filenames
  )
end

function BasicContParamIOStream(
  size::Tuple,
  n::Integer;
  monitor::Vector{Bool}=[true; fill(false, 12)],
  filepath::AbstractString="",
  filesuffix::AbstractString="csv",
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(BasicContParamIOStream)

  filenames = Array{AbstractString}(14)
  for i in 1:13
    filenames[i] = (monitor[i] == false ? "" : joinpath(filepath, string(fnames[i])*"."*filesuffix))
  end
  filenames[14] = (isempty(diagnostickeys) ? "" : joinpath(filepath, "diagnosticvalues."*filesuffix))

  BasicContParamIOStream(size, n, filenames, diagnostickeys, mode)
end

function BasicContParamIOStream(
  size::Tuple,
  n::Integer,
  monitor::Vector{Symbol};
  filepath::AbstractString="",
  filesuffix::AbstractString="csv",
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(BasicContParamIOStream)
  BasicContParamIOStream(
    size,
    n,
    monitor=[fnames[i] in monitor ? true : false for i in 1:13],
    filepath=filepath,
    filesuffix=filesuffix,
    diagnostickeys=diagnostickeys,
    mode=mode
  )
end

function open(iostream::BasicContParamIOStream, filenames::Vector{S}, mode::AbstractString="w") where {S<:AbstractString}
  iostream.names = filenames
  fnames = fieldnames(BasicContParamIOStream)

  for i in 1:14
    if getfield(iostream, fnames[i]) != nothing
      setfield!(iostream, fnames[i], open(filenames[i], mode))
    end
  end
end

function close(iostream::BasicContParamIOStream)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:14
    if getfield(iostream, fnames[i]) != nothing
      close(getfield(iostream, fnames[i]))
    end
  end
end

function mark(iostream::BasicContParamIOStream)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:14
    if getfield(iostream, fnames[i]) != nothing
      mark(getfield(iostream, fnames[i]))
    end
  end
end

function reset(iostream::BasicContParamIOStream)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:14
    if getfield(iostream, fnames[i]) != nothing
      reset(getfield(iostream, fnames[i]))
    end
  end
end

function flush(iostream::BasicContParamIOStream)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:14
    if getfield(iostream, fnames[i]) != nothing
      flush(getfield(iostream, fnames[i]))
    end
  end
end

function write(iostream::BasicContParamIOStream, state::ParameterState{Continuous, F}) where {F<:VariateForm}
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:14
    if getfield(iostream, fnames[i]) != nothing
      write(getfield(iostream, fnames[i]), join(getfield(state, fnames[i]), ','), "\n")
    end
  end
end

function write(iostream::BasicContParamIOStream, nstate::BasicContUnvParameterNState)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:13
    if getfield(iostream, fnames[i]) != nothing
      writedlm(getfield(iostream, fnames[i]), getfield(nstate, fnames[i]))
    end
  end
  if iostream.diagnosticvalues != nothing
    writedlm(iostream.diagnosticvalues, nstate.diagnosticvalues', ',')
  end
end

function write(iostream::BasicContParamIOStream, nstate::BasicContMuvParameterNState)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 2:4
    if getfield(iostream, fnames[i]) != nothing
      writedlm(getfield(iostream, fnames[i]), getfield(nstate, fnames[i]))
    end
  end
  for i in (1, 5, 6, 7, 14)
    if getfield(iostream, fnames[i]) != nothing
      writedlm(getfield(iostream, fnames[i]), getfield(nstate, fnames[i])', ',')
    end
  end
  for i in 8:10
    if getfield(iostream, fnames[i]) != nothing
      statelen = abs2(iostream.size)
      for j in 1:nstate.n
        write(iostream.stream, join(nstate.value[1+(j-1)*statelen:j*statelen], ','), "\n")
      end
    end
  end
  for i in 11:13
    if getfield(iostream, fnames[i]) != nothing
      statelen = iostream.size^3
      for j in 1:nstate.n
        write(iostream.stream, join(nstate.value[1+(j-1)*statelen:j*statelen], ','), "\n")
      end
    end
  end
end

function read!(iostream::BasicContParamIOStream, nstate::BasicContUnvParameterNState{N}) where N<:Real
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:13
    if getfield(iostream, fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(getfield(iostream, fnames[i]), ',', N)))
    end
  end
  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function read!(iostream::BasicContParamIOStream, nstate::BasicContMuvParameterNState{N}) where N<:Real
  fnames = fieldnames(BasicContParamIOStream)
  for i in 2:4
    if getfield(iostream, fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(getfield(iostream, fnames[i]), ',', N)))
    end
  end
  for i in (1, 5, 6, 7)
    if getfield(iostream, fnames[i]) != nothing
      setfield!(nstate, fnames[i], readdlm(getfield(iostream, fnames[i]), ',', N)')
    end
  end
  for i in 8:10
    if getfield(iostream, fnames[i]) != nothing
      statelen = abs2(iostream.size)
      line = 1
      while !eof(iostream.stream)
        nstate.value[1+(line-1)*statelen:line*statelen] =
          [parse(N, c) for c in split(chomp(readline(iostream.stream)), ',')]
        line += 1
      end
    end
  end
  for i in 11:13
    if getfield(iostream, fnames[i]) != nothing
      statelen = iostream.size^3
      line = 1
      while !eof(iostream.stream)
        nstate.value[1+(line-1)*statelen:line*statelen] =
          [parse(N, c) for c in split(chomp(readline(iostream.stream)), ',')]
        line += 1
      end
    end
  end
  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function read(iostream::BasicContParamIOStream, T::Type{N}) where N<:Real
  local nstate::ContinuousParameterNState
  fnames = fieldnames(BasicContParamIOStream)
  l = length(iostream.size)

  if l == 0
    nstate = BasicContUnvParameterNState(
      iostream.n,
      [getfield(iostream, fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      T
    )
  elseif l == 1
    nstate = BasicContMuvParameterNState(
      iostream.size[1],
      iostream.n,
      [getfield(iostream, fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      T
    )
  else
    error("BasicContParamIOStream.size must be a tuple of length 0 or 1, got $(iostream.size) length")
  end

  read!(iostream, nstate)

  nstate
end

function show(io::IO, iostream::BasicContParamIOStream)
  fnames = fieldnames(BasicContParamIOStream)
  fbool = map(n -> getfield(iostream, n) != nothing, fnames[1:13])
  indentation = "  "

  println(io, "BasicContParamIOStream:")

  println(io, indentation*"state size = $(iostream.size)")
  println(io, indentation*"number of states = $(iostream.n)")

  print(io, indentation*"monitored components:")
  if !any(fbool)
    println(io, " none")
  else
    print(io, "\n")
    for i in 1:13
      if fbool[i]
        println(io, string(indentation^2, fnames[i]))
      end
    end
  end

  print(io, indentation*"diagnostics:")
  if isempty(iostream.diagnostickeys)
    print(io, " none")
  else
    for k in iostream.diagnostickeys
      print(io, "\n")
      print(io, string(indentation^2, k))
    end
  end
end
