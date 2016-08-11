### BasicContParamIOStream

type BasicContParamIOStream <: ParameterIOStream{Continuous}
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
  open::Function
  close::Function
  mark::Function
  reset::Function
  flush::Function
  write::Function

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

    instance.open = eval(codegen(:open, instance, fnames))
    instance.close = eval(codegen(:close, instance, fnames))
    instance.mark = eval(codegen(:mark, instance, fnames))
    instance.reset = eval(codegen(:reset, instance, fnames))
    instance.flush = eval(codegen(:flush, instance, fnames))
    instance.write = eval(codegen(:write, instance, fnames))

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

  filenames = Array(AbstractString, 14)
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

function codegen(::Type{Val{:open}}, iostream::BasicContParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  push!(body,:(_iostream.names = _names))

  for i in 1:14
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(setfield!(_iostream, $(QuoteNode(f)), open(_names[$i], _mode))))
    end
  end

  @gensym _open

  quote
    function $_open{S<:AbstractString}(_iostream::BasicContParamIOStream, _names::Vector{S}, _mode::AbstractString="w")
      $(body...)
    end
  end
end

function codegen(::Type{Val{:close}}, iostream::BasicContParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:14
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(close(getfield(_iostream, $(QuoteNode(f))))))
    end
  end

  @gensym _close

  quote
    function $_close(_iostream::BasicContParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:mark}}, iostream::BasicContParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:14
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(mark(getfield(_iostream, $(QuoteNode(f))))))
    end
  end

  @gensym _mark

  quote
    function $_mark(_iostream::BasicContParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:reset}}, iostream::BasicContParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:14
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(reset(getfield($(iostream), $(QuoteNode(f))))))
    end
  end

  @gensym _reset

  quote
    function $_reset(_iostream::BasicContParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:flush}}, iostream::BasicContParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:14
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(flush(getfield($(iostream), $(QuoteNode(f))))))
    end
  end

  @gensym _flush

  quote
    function $_flush(_iostream::BasicContParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:write}}, iostream::BasicContParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted

  for i in 1:14
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(
        body,
        :(write(getfield($(iostream), $(QuoteNode(f))), join(getfield(_state, $(QuoteNode(f))), ','), "\n"))
      )
    end
  end

  @gensym _write

  quote
    function $_write{F<:VariateForm}(_iostream::BasicContParamIOStream, _state::ParameterState{Continuous, F})
      $(body...)
    end
  end
end

function Base.write(iostream::BasicContParamIOStream, nstate::BasicContUnvParameterNState)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i]))
    end
  end
  if iostream.diagnosticvalues != nothing
    writedlm(iostream.diagnosticvalues, nstate.diagnosticvalues', ',')
  end
end

function Base.write(iostream::BasicContParamIOStream, nstate::BasicContMuvParameterNState)
  fnames = fieldnames(BasicContParamIOStream)
  for i in 2:4
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i]))
    end
  end
  for i in (1, 5, 6, 7, 14)
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i])', ',')
    end
  end
  for i in 8:10
    if iostream.(fnames[i]) != nothing
      statelen = abs2(iostream.size)
      for j in 1:nstate.n
        write(iostream.stream, join(nstate.value[1+(j-1)*statelen:j*statelen], ','), "\n")
      end
    end
  end
  for i in 11:13
    if iostream.(fnames[i]) != nothing
      statelen = iostream.size^3
      for j in 1:nstate.n
        write(iostream.stream, join(nstate.value[1+(j-1)*statelen:j*statelen], ','), "\n")
      end
    end
  end
end

function Base.read!{N<:Real}(iostream::BasicContParamIOStream, nstate::BasicContUnvParameterNState{N})
  fnames = fieldnames(BasicContParamIOStream)
  for i in 1:13
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(iostream.(fnames[i]), ',', N)))
    end
  end
  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function Base.read!{N<:Real}(iostream::BasicContParamIOStream, nstate::BasicContMuvParameterNState{N})
  fnames = fieldnames(BasicContParamIOStream)
  for i in 2:4
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(iostream.(fnames[i]), ',', N)))
    end
  end
  for i in (1, 5, 6, 7)
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], readdlm(iostream.(fnames[i]), ',', N)')
    end
  end
  for i in 8:10
    if iostream.(fnames[i]) != nothing
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
    if iostream.(fnames[i]) != nothing
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

function Base.read{N<:Real}(iostream::BasicContParamIOStream, T::Type{N})
  nstate::ContinuousParameterNState
  fnames = fieldnames(BasicContParamIOStream)
  l = length(iostream.size)

  if l == 0
    nstate = BasicContUnvParameterNState(
      iostream.n,
      [iostream.(fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      T
    )
  elseif l == 1
    nstate = BasicContMuvParameterNState(
      iostream.size[1],
      iostream.n,
      [iostream.(fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      T
    )
  else
    error("BasicContParamIOStream.size must be a tuple of length 0 or 1, got $(iostream.size) length")
  end

  read!(iostream, nstate)

  nstate
end

function Base.show(io::IO, iostream::BasicContParamIOStream)
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
