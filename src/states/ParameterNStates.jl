### Abstract parameter NStates

abstract ParameterNState{S<:ValueSupport, F<:VariateForm} <: VariableNState{F}

typealias MarkovChain ParameterNState

diagnostics(nstate::ParameterNState) =
  Dict(zip(nstate.diagnostickeys, Any[nstate.diagnosticvalues[i, :][:] for i = 1:size(nstate.diagnosticvalues, 1)]))

### Parameter NState subtypes

## BasicContUnvParameterNState

type BasicContUnvParameterNState{N<:Real} <: ParameterNState{Continuous, Univariate}
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
  n::Int
  diagnostickeys::Vector{Symbol}
  copy::Function

  function BasicContUnvParameterNState(
    n::Int,
    monitor::Vector{Bool}=[true; fill(false, 12)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{N}=Float64,
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  )
    instance = new()

    l = Array(Int, 13)
    for i in 1:13
      l[i] = (monitor[i] == false ? zero(Int) : n)
    end

    fnames = fieldnames(BasicContUnvParameterNState)
    for i in 1:13
      setfield!(instance, fnames[i], Array(N, l[i]))
    end

    instance.diagnosticvalues = diagnosticvalues

    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance.copy = eval(codegen_copy_continuous_univariate_parameter_nstate(instance, monitor))

    instance
  end
end

BasicContUnvParameterNState{N<:Real}(
  n::Int,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) =
  BasicContUnvParameterNState{N}(n, monitor, diagnostickeys, N, diagnosticvalues)

function BasicContUnvParameterNState{N<:Real}(
  n::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
)
  fnames = fieldnames(BasicContUnvParameterNState)
  BasicContUnvParameterNState(n, [fnames[i] in monitor ? true : false for i in 1:13], diagnostickeys, N, diagnosticvalues)
end

typealias ContUnvMarkovChain BasicContUnvParameterNState

# To visually inspect code generation via codegen_copy_continuous_univariate_parameter_nstate, try for example
# using Lora
#
# nstate = ContUnvMarkovChain(Float64, 4)
# Lora.codegen_copy_continuous_univariate_parameter_nstate(nstate, [true; fill(false, 12)])

function codegen_copy_continuous_univariate_parameter_nstate(nstate::BasicContUnvParameterNState, monitor::Vector{Bool})
  body = []
  fnames = fieldnames(BasicContUnvParameterNState)
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted

  for j in 1:13
    if monitor[j]
      f = fnames[j]
      push!(body, :(getfield(_nstate, $(QuoteNode(f)))[_i] = getfield(_state, $(QuoteNode(f)))))
    end
  end

  if !isempty(nstate.diagnosticvalues)
    push!(body, :(_nstate.diagnosticvalues[:, _i] = _state.diagnosticvalues))
  end

  @gensym copy_continuous_univariate_parameter_nstate

  quote
    function $copy_continuous_univariate_parameter_nstate(
      _nstate::BasicContUnvParameterNState,
      _state::BasicContUnvParameterState,
      _i::Int
    )
      $(body...)
    end
  end
end

Base.copy!(nstate::BasicContUnvParameterNState, state::BasicContUnvParameterState, i::Int) = nstate.copy(nstate, state, i)

Base.eltype{N<:Real}(::Type{BasicContUnvParameterNState{N}}) = N
Base.eltype{N<:Real}(s::BasicContUnvParameterNState{N}) = N

Base.(:(==)){S<:BasicContUnvParameterNState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:16]])

Base.isequal{S<:BasicContUnvParameterNState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:16]])

function Base.show{N<:Real}(io::IO, nstate::BasicContUnvParameterNState{N})
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

Base.writemime{N<:Real}(io::IO, ::MIME"text/plain", nstate::BasicContUnvParameterNState{N}) = show(io, nstate)

## BasicContMuvParameterNState

type BasicContMuvParameterNState{N<:Real} <: ParameterNState{Continuous, Multivariate}
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
  size::Int
  n::Int
  diagnostickeys::Vector{Symbol}
  copy::Function

  function BasicContMuvParameterNState(
    size::Int,
    n::Int,
    monitor::Vector{Bool}=[true; fill(false, 12)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{N}=Float64,
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  )
    instance = new()

    fnames = fieldnames(BasicContMuvParameterNState)
    for i in 2:4
      l = (monitor[i] == false ? zero(Int) : n)
      setfield!(instance, fnames[i], Array(N, l))
    end
    for i in (1, 5, 6, 7)
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, fnames[i], Array(N, s, l))
    end
    for i in 8:10
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, fnames[i], Array(N, s, s, l))
    end
    for i in 11:13
      s, l = (monitor[i] == false ? (zero(Int), zero(Int)) : (size, n))
      setfield!(instance, fnames[i], Array(N, s, s, s, l))
    end

    instance.diagnosticvalues = diagnosticvalues

    instance.size = size
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance.copy = eval(codegen_copy_continuous_multivariate_parameter_nstate(instance, monitor))

    instance
  end
end

BasicContMuvParameterNState{N<:Real}(
  size::Int,
  n::Int,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) =
  BasicContMuvParameterNState{N}(size, n, monitor, diagnostickeys, N, diagnosticvalues)

function BasicContMuvParameterNState{N<:Real}(
  size::Int,
  n::Int,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
)
  fnames = fieldnames(BasicContMuvParameterNState)
  BasicContMuvParameterNState(
    size, n, [fnames[i] in monitor ? true : false for i in 1:13], diagnostickeys, N, diagnosticvalues
  )
end

typealias ContMuvMarkovChain BasicContMuvParameterNState

# To visually inspect code generation via codegen_copy_continuous_multivariate_parameter_nstate, try for example
# using Lora
#
# nstate = ContMuvMarkovChain(Float64, 2, 4)
# Lora.codegen_copy_continuous_multivariate_parameter_nstate(nstate, [true; fill(false, 12)])

function codegen_copy_continuous_multivariate_parameter_nstate(
  nstate::BasicContMuvParameterNState,
  monitor::Vector{Bool}
)
  body = []
  fnames = fieldnames(BasicContMuvParameterNState)
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted
  statelen::Int

  for j in 2:4
    if monitor[j]
      f = fnames[j]
      push!(body, :(getfield(_nstate, $(QuoteNode(f)))[_i] = getfield(_state, $(QuoteNode(f)))))
    end
  end

  for j in (1, 5, 6, 7)
    if monitor[j]
      f = fnames[j]
      push!(
        body,
        :(getfield(_nstate, $(QuoteNode(f)))[1+(_i-1)*_state.size:_i*_state.size] = getfield(_state, $(QuoteNode(f))))
      )
    end
  end

  if monitor[8] || monitor[9] || monitor[10]
    statelen = (nstate.size)^2
  end
  for j in 8:10
    if monitor[j]
      f = fnames[j]
      push!(
        body,
        :(getfield(_nstate, $(QuoteNode(f)))[1+(_i-1)*$(statelen):_i*$(statelen)] = getfield(_state, $(QuoteNode(f))))
      )
    end
  end

  if monitor[11] || monitor[12] || monitor[13]
    statelen = (nstate.size)^3
  end
  for j in 11:13
    if monitor[j]
      f = fnames[j]
      push!(
        body,
        :(getfield(_nstate, $(QuoteNode(f)))[1+(_i-1)*$(statelen):_i*$(statelen)] = getfield(_state, $(QuoteNode(f))))
      )
    end
  end

  if !isempty(nstate.diagnosticvalues)
    push!(body, :(_nstate.diagnosticvalues[:, _i] = _state.diagnosticvalues))
  end

  @gensym copy_continuous_multivariate_parameter_nstate

  quote
    function $copy_continuous_multivariate_parameter_nstate(
      _nstate::BasicContMuvParameterNState,
      _state::BasicContMuvParameterState,
      _i::Int
    )
      $(body...)
    end
  end
end

Base.copy!(nstate::BasicContMuvParameterNState, state::BasicContMuvParameterState, i::Int) = nstate.copy(nstate, state, i)

Base.eltype{N<:Real}(::Type{BasicContMuvParameterNState{N}}) = N
Base.eltype{N<:Real}(s::BasicContMuvParameterNState{N}) = N

Base.(:(==)){S<:BasicContMuvParameterNState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:17]])

Base.isequal{S<:BasicContMuvParameterNState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:17]])

function Base.show{N<:Real}(io::IO, nstate::BasicContMuvParameterNState{N})
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

Base.writemime{N<:Real}(io::IO, ::MIME"text/plain", nstate::BasicContMuvParameterNState{N}) = show(io, nstate)
