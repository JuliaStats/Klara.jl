mutable struct BasicDiscMuvParameterNState{NI<:Integer, NR<:Real} <: ParameterNState{Discrete, Multivariate}
  value::Matrix{NI}
  loglikelihood::Vector{NR}
  logprior::Vector{NR}
  logtarget::Vector{NR}
  diagnosticvalues::Matrix
  size::Integer
  sizesquared::Integer
  sizecubed::Integer
  monitor::Vector{Bool}
  n::Integer
  diagnostickeys::Vector{Symbol}

  function BasicDiscMuvParameterNState{NI, NR}(
    size::Integer,
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 3)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{NI}=Int64,
    ::Type{NR}=Float64,
    diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  ) where {NI<:Integer, NR<:Real}
    instance = new()

    fnames = fieldnames(BasicDiscMuvParameterNState)
    setfield!(instance, fnames[1], Array{NI}((monitor[1] == false ? (zero(Integer), zero(Integer)) : (size, n))...))
    for i in 2:4
      l = (monitor[i] == false ? zero(Integer) : n)
      setfield!(instance, fnames[i], Array{NR}(l))
    end

    instance.diagnosticvalues = diagnosticvalues
    instance.size = size
    instance.sizesquared = instance.size^2
    instance.sizecubed = instance.size^3
    instance.monitor = monitor
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance
  end
end

BasicDiscMuvParameterNState(
  size::Integer,
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 3)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where {NI<:Integer, NR<:Real} =
  BasicDiscMuvParameterNState{NI, NR}(size, n, monitor, diagnostickeys, NI, NR, diagnosticvalues)

function BasicDiscMuvParameterNState(
  size::Integer,
  n::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where {NI<:Integer, NR<:Real}
  fnames = fieldnames(BasicDiscMuvParameterNState)
  BasicDiscMuvParameterNState(
    size, n, [fnames[i] in monitor ? true : false for i in 1:4], diagnostickeys, NI, NR, diagnosticvalues
  )
end

const DiscMuvMarkovChain = BasicDiscMuvParameterNState

function copy!(nstate::BasicDiscMuvParameterNState, state::BasicDiscMuvParameterState, i::Integer)
  fnames = fieldnames(BasicDiscMuvParameterNState)

  if nstate.monitor[1]
    getfield(nstate, fnames[1])[1+(i-1)*state.size:i*state.size] = getfield(state, fnames[1])
  end

  for j in 2:4
    if nstate.monitor[j]
      getfield(nstate, fnames[j])[i] = getfield(state, fnames[j])
    end
  end

  if !isempty(nstate.diagnosticvalues)
    nstate.diagnosticvalues[:, i] = state.diagnosticvalues
  end
end

eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscMuvParameterNState{NI, NR}}) = (NI, NR)
eltype{NI<:Integer, NR<:Real}(::BasicDiscMuvParameterNState{NI, NR}) = (NI, NR)

==(z::S, w::S) where {S<:BasicDiscMuvParameterNState} = reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:8]])

isequal(z::S, w::S) where {S<:BasicDiscMuvParameterNState} =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:8]])

function show(io::IO, nstate::BasicDiscMuvParameterNState{NI, NR}) where {NI<:Integer, NR<:Real}
  fnames = fieldnames(BasicDiscMuvParameterNState)
  fbool = map(n -> !isempty(getfield(nstate, n)), fnames[1:4])
  indentation = "  "

  println(io, "BasicDiscMuvParameterNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"state size = $(nstate.size)")
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
