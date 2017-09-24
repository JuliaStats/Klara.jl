mutable struct BasicContUnvParameterNState{N<:Real} <: ParameterNState{Continuous, Univariate}
  value::Vector{N}
  loglikelihood::Vector{N}
  logprior::Vector{N}
  logtarget::Vector{N}
  gradloglikelihood::Vector{N}
  gradlogprior::Vector{N}
  gradlogtarget::Vector{N}
  tensorloglikelihood::Vector{N}
  tensorlogprior::Vector{N}
  tensorlogtarget::Vector{N}
  dtensorloglikelihood::Vector{N}
  dtensorlogprior::Vector{N}
  dtensorlogtarget::Vector{N}
  diagnosticvalues::Matrix
  monitor::Vector{Bool}
  n::Integer
  diagnostickeys::Vector{Symbol}

  function BasicContUnvParameterNState{N}(
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 12)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{N}=Float64,
    diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  ) where N<:Real
    instance = new()

    l = Array{Integer}(13)
    for i in 1:13
      l[i] = (monitor[i] == false ? zero(Integer) : n)
    end

    fnames = fieldnames(BasicContUnvParameterNState)
    for i in 1:13
      setfield!(instance, fnames[i], Array{N}(l[i]))
    end

    instance.diagnosticvalues = diagnosticvalues
    instance.monitor = monitor
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance
  end
end

BasicContUnvParameterNState(
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where N<:Real =
  BasicContUnvParameterNState{N}(n, monitor, diagnostickeys, N, diagnosticvalues)

function BasicContUnvParameterNState(
  n::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where N<:Real
  fnames = fieldnames(BasicContUnvParameterNState)
  BasicContUnvParameterNState(n, [fnames[i] in monitor ? true : false for i in 1:13], diagnostickeys, N, diagnosticvalues)
end

const ContUnvMarkovChain = BasicContUnvParameterNState

function copy!(nstate::BasicContUnvParameterNState, state::BasicContUnvParameterState, i::Integer)
  fnames = fieldnames(BasicContUnvParameterNState)

  for j in 1:13
    if nstate.monitor[j]
      getfield(nstate, fnames[j])[i] = getfield(state, fnames[j])
    end
  end

  if !isempty(nstate.diagnosticvalues)
    nstate.diagnosticvalues[:, i] = state.diagnosticvalues
  end
end

eltype{N<:Real}(::Type{BasicContUnvParameterNState{N}}) = N
eltype{N<:Real}(::BasicContUnvParameterNState{N}) = N

==(z::S, w::S) where {S<:BasicContUnvParameterNState} = reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:16]])

isequal(z::S, w::S) where {S<:BasicContUnvParameterNState} =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:16]])

function show(io::IO, nstate::BasicContUnvParameterNState{N}) where N<:Real
  fnames = fieldnames(BasicContUnvParameterNState)
  fbool = map(n -> !isempty(getfield(nstate, n)), fnames[1:13])
  indentation = "  "

  println(io, "BasicContUnvParameterNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"number of states = $(nstate.n)")

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
  if isempty(nstate.diagnostickeys)
    print(io, " none")
  else
    for k in nstate.diagnostickeys
      print(io, "\n")
      print(io, string(indentation^2, k))
    end
  end
end
