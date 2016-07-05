### BasicDiscParamIOStream

type BasicDiscParamIOStream <: ParameterIOStream{Discrete}
  value::Union{IOStream, Void}
  loglikelihood::Union{IOStream, Void}
  logprior::Union{IOStream, Void}
  logtarget::Union{IOStream, Void}
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

    instance.open = eval(codegen(:open, instance, fnames))
    instance.close = eval(codegen(:close, instance, fnames))
    instance.mark = eval(codegen(:mark, instance, fnames))
    instance.reset = eval(codegen(:reset, instance, fnames))
    instance.flush = eval(codegen(:flush, instance, fnames))
    instance.write = eval(codegen(:write, instance, fnames))

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

  filenames = Array(AbstractString, 5)
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

function codegen(::Type{Val{:open}}, iostream::BasicDiscParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  push!(body,:(_iostream.names = _names))

  for i in 1:5
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(setfield!(_iostream, $(QuoteNode(f)), open(_names[$i], _mode))))
    end
  end

  @gensym _open

  quote
    function $_open{S<:AbstractString}(_iostream::BasicDiscParamIOStream, _names::Vector{S}, _mode::AbstractString="w")
      $(body...)
    end
  end
end

function codegen(::Type{Val{:close}}, iostream::BasicDiscParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:5
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(close(getfield(_iostream, $(QuoteNode(f))))))
    end
  end

  @gensym _close

  quote
    function $_close(_iostream::BasicDiscParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:mark}}, iostream::BasicDiscParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:5
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(mark(getfield(_iostream, $(QuoteNode(f))))))
    end
  end

  @gensym _mark

  quote
    function $_mark(_iostream::BasicDiscParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:reset}}, iostream::BasicDiscParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:5
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(reset(getfield($(iostream), $(QuoteNode(f))))))
    end
  end

  @gensym _reset

  quote
    function $_reset(_iostream::BasicDiscParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:flush}}, iostream::BasicDiscParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:5
    if iostream.(fnames[i]) != nothing
      f = fnames[i]
      push!(body, :(flush(getfield($(iostream), $(QuoteNode(f))))))
    end
  end

  @gensym _flush

  quote
    function $_flush(_iostream::BasicDiscParamIOStream)
      $(body...)
    end
  end
end

function codegen(::Type{Val{:write}}, iostream::BasicDiscParamIOStream, fnames::Vector{Symbol})
  body = []
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted

  for i in 1:5
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
    function $_write{F<:VariateForm}(_iostream::BasicDiscParamIOStream, _state::ParameterState{Discrete, F})
      $(body...)
    end
  end
end

function Base.write(iostream::BasicDiscParamIOStream, nstate::BasicDiscUnvParameterNState)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:4
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i]))
    end
  end
  if iostream.diagnosticvalues != nothing
    writedlm(iostream.diagnosticvalues, nstate.diagnosticvalues', ',')
  end
end

function Base.write(iostream::BasicDiscParamIOStream, nstate::BasicDiscMuvParameterNState)
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 2:4
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i]))
    end
  end
  for i in (1, 5)
    if iostream.(fnames[i]) != nothing
      writedlm(iostream.(fnames[i]), nstate.(fnames[i])', ',')
    end
  end
end

function Base.read!{NI<:Integer, NR<:Real}(iostream::BasicDiscParamIOStream, nstate::BasicDiscUnvParameterNState{NI, NR})
  t = [NI, fill(NR, 3)]
  fnames = fieldnames(BasicDiscParamIOStream)
  for i in 1:4
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(iostream.(fnames[i]), ',', t[i])))
    end
  end
  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function Base.read!{NI<:Integer, NR<:Real}(iostream::BasicDiscParamIOStream, nstate::BasicDiscMuvParameterNState{NI, NR})
  fnames = fieldnames(BasicDiscParamIOStream)
  if iostream.(fnames[1]) != nothing
    setfield!(nstate, fnames[1], readdlm(iostream.(fnames[1]), ',', NI)')
  end
  for i in 2:4
    if iostream.(fnames[i]) != nothing
      setfield!(nstate, fnames[i], vec(readdlm(iostream.(fnames[i]), ',', NR)))
    end
  end
  if iostream.diagnosticvalues != nothing
    nstate.diagnosticvalues = readdlm(iostream.diagnosticvalues, ',', Any)'
  end
end

function Base.read{NI<:Integer, NR<:Real}(iostream::BasicDiscParamIOStream, TI::Type{NI}, TR::Type{NR})
  nstate::DiscreteParameterNState
  fnames = fieldnames(BasicDiscParamIOStream)
  l = length(iostream.size)

  if l == 0
    nstate = BasicDiscUnvParameterNState(
      iostream.n,
      [iostream.(fnames[i]) != nothing ? true : false for i in 1:13],
      iostream.diagnostickeys,
      TI,
      TR
    )
  elseif l == 1
    nstate = BasicDiscMuvParameterNState(
      iostream.size[1],
      iostream.n,
      [iostream.(fnames[i]) != nothing ? true : false for i in 1:13],
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

function Base.show(io::IO, iostream::BasicDiscParamIOStream)
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
