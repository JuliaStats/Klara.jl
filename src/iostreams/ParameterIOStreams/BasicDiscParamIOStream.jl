### BasicDiscParamIOStream

mutable struct BasicDiscParamIOStream <: ParameterIOStream{Discrete}
  value::Union{IOStream, Void}
  loglikelihood::Union{IOStream, Void}
  logprior::Union{IOStream, Void}
  logtarget::Union{IOStream, Void}
  diagnosticvalues::Union{IOStream, Void}
  names::Vector{AbstractString}
  size::Tuple
  n::Integer
  diagnostickeys::Vector{Symbol}

  function BasicDiscParamIOStream(
    size::Tuple,
    n::Integer,
    streams::Vector{Union{IOStream, Void}},
    diagnostickeys::Vector{Symbol}=Symbol[],
    filenames::Vector{AbstractString}=[(streams[i] == nothing) ? "" : streams[i].name[7:end-1] for i in 1:5]
  )
    instance = new()

    fnames = fieldnames(BasicDiscParamIOStream)
    for i in 1:5
      setfield!(instance, fnames[i], streams[i])
    end

    instance.names = filenames

    instance.size = size
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance
  end
end

function BasicDiscParamIOStream(
  size::Tuple,
  n::Integer,
  filenames::Vector{AbstractString},
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(BasicDiscParamIOStream)
  BasicDiscParamIOStream(
    size,
    n,
    [isempty(filenames[i]) ? nothing : open(filenames[i], mode) for i in 1:5],
    diagnostickeys,
    filenames
  )
end

function BasicDiscParamIOStream(
  size::Tuple,
  n::Integer;
  monitor::Vector{Bool}=[true; fill(false, 3)],
  filepath::AbstractString="",
  filesuffix::AbstractString="csv",
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(BasicDiscParamIOStream)

  filenames = Array{AbstractString}(5)
  for i in 1:4
    filenames[i] = (monitor[i] == false ? "" : joinpath(filepath, string(fnames[i])*"."*filesuffix))
  end
  filenames[5] = (isempty(diagnostickeys) ? "" : joinpath(filepath, "diagnosticvalues."*filesuffix))

  BasicDiscParamIOStream(size, n, filenames, diagnostickeys, mode)
end

function BasicDiscParamIOStream(
  size::Tuple,
  n::Integer,
  monitor::Vector{Symbol};
  filepath::AbstractString="",
  filesuffix::AbstractString="csv",
  diagnostickeys::Vector{Symbol}=Symbol[],
  mode::AbstractString="w"
)
  fnames = fieldnames(BasicDiscParamIOStream)
  BasicDiscParamIOStream(
    size,
    n,
    monitor=[fnames[i] in monitor ? true : false for i in 1:4],
    filepath=filepath,
    filesuffix=filesuffix,
    diagnostickeys=diagnostickeys,
    mode=mode
  )
end

function open(iostream::BasicDiscParamIOStream, filenames::Vector{S}, mode::AbstractString="w") where {S<:AbstractString}
  iostream.names = filenames
  fnames = fieldnames(BasicDiscParamIOStream)

  for i in 1:5
    if getfield(iostream, fnames[i]) != nothing
      setfield!(iostream, fnames[i], open(filenames[i], mode))
    end
  end
end

function close(iostream::BasicDiscParamIOStream)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:5
    if getfield(iostream, fnames[i]) != nothing
      close(getfield(iostream, fnames[i]))
    end
  end
end

function mark(iostream::BasicDiscParamIOStream)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:5
    if getfield(iostream, fnames[i]) != nothing
      mark(getfield(iostream, fnames[i]))
    end
  end
end

function reset(iostream::BasicDiscParamIOStream)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:5
    if getfield(iostream, fnames[i]) != nothing
      reset(getfield(iostream, fnames[i]))
    end
  end
end

function flush(iostream::BasicDiscParamIOStream)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:5
    if getfield(iostream, fnames[i]) != nothing
      flush(getfield(iostream, fnames[i]))
    end
  end
end

function write(iostream::BasicDiscParamIOStream, state::ParameterState{Discrete, F}) where {F<:VariateForm}
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:5
    if getfield(iostream, fnames[i]) != nothing
      write(getfield(iostream, fnames[i]), join(getfield(state, fnames[i]), ','), "\n")
    end
  end
end

function write(iostream::BasicDiscParamIOStream, nstate::BasicDiscUnvParameterNState)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:4
    if getfield(iostream, fnames[i]) != nothing
      writedlm(getfield(iostream, fnames[i]), nstate.(fnames[i]))
    end
  end
  if iostream.diagnosticvalues != nothing
    writedlm(iostream.diagnosticvalues, nstate.diagnosticvalues', ',')
  end
end

function write(iostream::BasicDiscParamIOStream, nstate::BasicDiscMuvParameterNState)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 2:4
    if getfield(iostream, fnames[i]) != nothing
      writedlm(getfield(iostream, fnames[i]), nstate.(fnames[i]))
    end
  end
  for i in (1, 5)
    if getfield(iostream, fnames[i]) != nothing
      writedlm(getfield(iostream, fnames[i]), nstate.(fnames[i])', ',')
    end
  end
end

function read!(iostream::BasicDiscParamIOStream, nstate::BasicDiscUnvParameterNState{NI, NR}) where {NI<:Integer, NR<:Real}
  t = [NI, fill(NR, 3)]
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:4
    if getfield(iostream, fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(getfield(iostream, fnames[i]), ',', t[i])))
    end
  end
  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function read!(iostream::BasicDiscParamIOStream, nstate::BasicDiscMuvParameterNState{NI, NR}) where {NI<:Integer, NR<:Real}
  fnames = fieldnames(BasicDiscParamIOStream)
  if getfield(iostream, fnames[1]) != nothing
    setfield!(nstate, fnames[1], readdlm(getfield(iostream, fnames[1]), ',', NI)')
  end
  for i in 2:4
    if getfield(iostream, fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(getfield(iostream, fnames[i]), ',', NR)))
    end
  end
  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function read(iostream::BasicDiscParamIOStream, TI::Type{NI}, TR::Type{NR}) where {NI<:Integer, NR<:Real}
  local nstate::DiscreteParameterNState
  fnames = fieldnames(BasicDiscParamIOStream)
  l = length(iostream.size)

  if l == 0
    nstate = BasicDiscUnvParameterNState(
      iostream.n,
      [getfield(iostream, fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      TI,
      TR
    )
  elseif l == 1
    nstate = BasicDiscMuvParameterNState(
      iostream.size[1],
      iostream.n,
      [getfield(iostream, fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      TI,
      TR
    )
  else
    error("BasicDiscParamIOStream.size must be a tuple of length 0 or 1, got $(iostream.size) length")
  end

  read!(iostream, nstate)

  nstate
end

function show(io::IO, iostream::BasicDiscParamIOStream)
  fnames = fieldnames(BasicDiscParamIOStream)
  fbool = map(n -> getfield(iostream, n) != nothing, fnames[1:4])
  indentation = "  "

  println(io, "BasicDiscParamIOStream:")

  println(io, indentation*"state size = $(iostream.size)")
  println(io, indentation*"number of states = $(iostream.n)")

  print(io, indentation*"monitored components:")
  if !any(fbool)
    println(io, " none")
  else
    print(io, "\n")
    for i in 1:4
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
