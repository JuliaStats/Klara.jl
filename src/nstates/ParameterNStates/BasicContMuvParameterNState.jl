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
  size::Integer
  n::Integer
  diagnostickeys::Vector{Symbol}
  copy::Function

  function BasicContMuvParameterNState(
    size::Integer,
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 12)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{N}=Float64,
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  )
    instance = new()

    fnames = fieldnames(BasicContMuvParameterNState)
    for i in 2:4
      l = (monitor[i] == false ? zero(Integer) : n)
      setfield!(instance, fnames[i], Array(N, l))
    end
    for i in (1, 5, 6, 7)
      s, l = (monitor[i] == false ? (zero(Integer), zero(Integer)) : (size, n))
      setfield!(instance, fnames[i], Array(N, s, l))
    end
    for i in 8:10
      s, l = (monitor[i] == false ? (zero(Integer), zero(Integer)) : (size, n))
      setfield!(instance, fnames[i], Array(N, s, s, l))
    end
    for i in 11:13
      s, l = (monitor[i] == false ? (zero(Integer), zero(Integer)) : (size, n))
      setfield!(instance, fnames[i], Array(N, s, s, s, l))
    end

    instance.diagnosticvalues = diagnosticvalues

    instance.size = size
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance.copy = eval(codegen(:copy, instance, monitor))

    instance
  end
end

BasicContMuvParameterNState{N<:Real}(
  size::Integer,
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) =
  BasicContMuvParameterNState{N}(size, n, monitor, diagnostickeys, N, diagnosticvalues)

function BasicContMuvParameterNState{N<:Real}(
  size::Integer,
  n::Integer,
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

codegen(f::Symbol, nstate::BasicContMuvParameterNState, monitor::Vector{Bool}) = codegen(Val{f}, nstate, monitor)

function codegen(::Type{Val{:copy}}, nstate::BasicContMuvParameterNState, monitor::Vector{Bool})
  body = []
  fnames = fieldnames(BasicContMuvParameterNState)
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted
  statelen::Integer

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

  @gensym _copy

  quote
    function $_copy(_nstate::BasicContMuvParameterNState, _state::BasicContMuvParameterState, _i::Integer)
      $(body...)
    end
  end
end

Base.eltype{N<:Real}(::Type{BasicContMuvParameterNState{N}}) = N
Base.eltype{N<:Real}(::BasicContMuvParameterNState{N}) = N

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
