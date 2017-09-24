mutable struct BasicDiscUnvParameterNState{NI<:Integer, NR<:Real} <: ParameterNState{Discrete, Univariate}
  value::Vector{NI}
  loglikelihood::Vector{NR}
  logprior::Vector{NR}
  logtarget::Vector{NR}
  diagnosticvalues::Matrix
  monitor::Vector{Bool}
  n::Integer
  diagnostickeys::Vector{Symbol}

  function BasicDiscUnvParameterNState{NI, NR}(
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 3)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{NI}=Int64,
    ::Type{NR}=Float64,
    diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  ) where {NI<:Integer, NR<:Real}
    instance = new()

    t = [NI; fill(NR, 3)]

    l = Array{Integer}(4)
    for i in 1:4
      l[i] = (monitor[i] == false ? zero(Integer) : n)
    end

    fnames = fieldnames(BasicDiscUnvParameterNState)
    for i in 1:4
      setfield!(instance, fnames[i], Array{t[i]}(l[i]))
    end

    instance.diagnosticvalues = diagnosticvalues
    instance.monitor = monitor
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance
  end
end

BasicDiscUnvParameterNState(
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 3)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where {NI<:Integer, NR<:Real} =
  BasicDiscUnvParameterNState{NI, NR}(n, monitor, diagnostickeys, NI, NR, diagnosticvalues)

function BasicDiscUnvParameterNState(
  n::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where {NI<:Integer, NR<:Real}
  fnames = fieldnames(BasicDiscUnvParameterNState)
  BasicDiscUnvParameterNState(
    n, [fnames[i] in monitor ? true : false for i in 1:4], diagnostickeys, NI, NR, diagnosticvalues
  )
end

const DiscUnvMarkovChain = BasicDiscUnvParameterNState

function copy!(nstate::BasicDiscUnvParameterNState, state::BasicDiscUnvParameterState, i::Integer)
  fnames = fieldnames(BasicDiscUnvParameterNState)

  for j in 1:4
    if nstate.monitor[j]
      getfield(nstate, fnames[j])[i] = getfield(state, fnames[j])
    end
  end

  if !isempty(nstate.diagnosticvalues)
    nstate.diagnosticvalues[:, i] = state.diagnosticvalues
  end
end

eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscUnvParameterNState{NI, NR}}) = (NI, NR)
eltype{NI<:Integer, NR<:Real}(::BasicDiscUnvParameterNState{NI, NR}) = (NI, NR)

==(z::S, w::S) where {S<:BasicDiscUnvParameterNState} = reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:7]])

isequal(z::S, w::S) where {S<:BasicDiscUnvParameterNState} =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:7]])

function show(io::IO, nstate::BasicDiscUnvParameterNState{NI, NR}) where {NI<:Integer, NR<:Real}
  fnames = fieldnames(BasicDiscUnvParameterNState)
  fbool = map(n -> !isempty(getfield(nstate, n)), fnames[1:4])
  indentation = "  "

  println(io, "BasicDiscUnvParameterNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"number of states = $(nstate.n)")

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
  if isempty(nstate.diagnostickeys)
    print(io, " none")
  else
    for k in nstate.diagnostickeys
      print(io, "\n")
      print(io, string(indentation^2, k))
    end
  end
end
