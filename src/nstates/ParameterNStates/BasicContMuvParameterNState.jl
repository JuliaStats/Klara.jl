mutable struct BasicContMuvParameterNState{N<:Real} <: ParameterNState{Continuous, Multivariate}
  value::Matrix{N}
  loglikelihood::Vector{N}
  logprior::Vector{N}
  logtarget::Vector{N}
  gradloglikelihood::Matrix{N}
  gradlogprior::Matrix{N}
  gradlogtarget::Matrix{N}
  tensorloglikelihood::Array{N, 3}
  tensorlogprior::Array{N, 3}
  tensorlogtarget::Array{N, 3}
  dtensorloglikelihood::Array{N, 4}
  dtensorlogprior::Array{N, 4}
  dtensorlogtarget::Array{N, 4}
  diagnosticvalues::Matrix
  size::Integer
  sizesquared::Integer
  sizecubed::Integer
  monitor::Vector{Bool}
  n::Integer
  diagnostickeys::Vector{Symbol}

  function BasicContMuvParameterNState{N}(
    size::Integer,
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 12)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{N}=Float64,
    diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  ) where N<:Real
    instance = new()

    fnames = fieldnames(BasicContMuvParameterNState)
    for i in 2:4
      l = (monitor[i] == false ? zero(Integer) : n)
      setfield!(instance, fnames[i], Array{N}(l))
    end
    for i in (1, 5, 6, 7)
      s, l = (monitor[i] == false ? (zero(Integer), zero(Integer)) : (size, n))
      setfield!(instance, fnames[i], Array{N}(s, l))
    end
    for i in 8:10
      s, l = (monitor[i] == false ? (zero(Integer), zero(Integer)) : (size, n))
      setfield!(instance, fnames[i], Array{N}(s, s, l))
    end
    for i in 11:13
      s, l = (monitor[i] == false ? (zero(Integer), zero(Integer)) : (size, n))
      setfield!(instance, fnames[i], Array{N}(s, s, s, l))
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

BasicContMuvParameterNState(
  size::Integer,
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where N<:Real =
  BasicContMuvParameterNState{N}(size, n, monitor, diagnostickeys, N, diagnosticvalues)

function BasicContMuvParameterNState(
  size::Integer,
  n::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array{Any}(length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) where N<:Real
  fnames = fieldnames(BasicContMuvParameterNState)
  BasicContMuvParameterNState(
    size, n, [fnames[i] in monitor ? true : false for i in 1:13], diagnostickeys, N, diagnosticvalues
  )
end

const ContMuvMarkovChain = BasicContMuvParameterNState

function copy!(nstate::BasicContMuvParameterNState, state::BasicContMuvParameterState, i::Integer)
  fnames = fieldnames(BasicContMuvParameterNState)

  for j in 2:4
    if nstate.monitor[j]
      getfield(nstate, fnames[j])[i] = getfield(state, fnames[j])
    end
  end

  for j in (1, 5, 6, 7)
    if nstate.monitor[j]
      getfield(nstate, fnames[j])[1+(i-1)*state.size:i*state.size] = getfield(state, fnames[j])
    end
  end

  for j in 8:10
    if nstate.monitor[j]
      getfield(nstate, fnames[j])[1+(i-1)*nstate.sizesquared:i*nstate.sizesquared] = getfield(state, fnames[j])
    end
  end

  for j in 11:13
    if nstate.monitor[j]
      getfield(nstate, fnames[j])[1+(i-1)*nstate.sizecubed:i*nstate.sizecubed] = getfield(state, fnames[j])
    end
  end

  if !isempty(nstate.diagnosticvalues)
    nstate.diagnosticvalues[:, i] = state.diagnosticvalues
  end
end

eltype{N<:Real}(::Type{BasicContMuvParameterNState{N}}) = N
eltype{N<:Real}(::BasicContMuvParameterNState{N}) = N

==(z::S, w::S) where {S<:BasicContMuvParameterNState} = reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:17]])

isequal(z::S, w::S) where {S<:BasicContMuvParameterNState} =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:17]])

function show(io::IO, nstate::BasicContMuvParameterNState{N}) where N<:Real
  fnames = fieldnames(BasicContMuvParameterNState)
  fbool = map(n -> !isempty(getfield(nstate, n)), fnames[1:13])
  indentation = "  "

  println(io, "BasicContMuvParameterNState:")

  println(io, indentation*"eltype: $(eltype(nstate))")
  println(io, indentation*"state size = $(nstate.size)")
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
