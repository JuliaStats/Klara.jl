type BasicDiscMuvParameterNState{NI<:Integer, NR<:Real} <: ParameterNState{Continuous, Multivariate}
  value::Matrix{NI}
  loglikelihood::Vector{NR}
  logprior::Vector{NR}
  logtarget::Vector{NR}
  diagnosticvalues::Matrix
  size::Integer
  n::Integer
  diagnostickeys::Vector{Symbol}
  copy::Function

  function BasicDiscMuvParameterNState(
    size::Integer,
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 3)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{NI}=Int64,
    ::Type{NR}=Float64,
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  )
    instance = new()

    fnames = fieldnames(BasicDiscMuvParameterNState)
    setfield!(instance, fnames[1], Array(NI, (monitor[1] == false ? (zero(Integer), zero(Integer)) : (size, n))...))
    for i in 2:4
      l = (monitor[i] == false ? zero(Integer) : n)
      setfield!(instance, fnames[i], Array(NR, l))
    end

    instance.diagnosticvalues = diagnosticvalues

    instance.size = size
    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance.copy = eval(codegen(:copy, instance, monitor))

    instance
  end
end

BasicDiscMuvParameterNState{NI<:Integer, NR<:Real}(
  size::Integer,
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 3)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) =
  BasicDiscMuvParameterNState{NI, NR}(size, n, monitor, diagnostickeys, NI, NR, diagnosticvalues)

function BasicDiscMuvParameterNState{NI<:Integer, NR<:Real}(
  size::Integer,
  n::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
)
  fnames = fieldnames(BasicDiscMuvParameterNState)
  BasicDiscMuvParameterNState(
    size, n, [fnames[i] in monitor ? true : false for i in 1:4], diagnostickeys, NI, NR, diagnosticvalues
  )
end

typealias DiscMuvMarkovChain BasicDiscMuvParameterNState

codegen(f::Symbol, nstate::BasicDiscMuvParameterNState, monitor::Vector{Bool}) = codegen(Val{f}, nstate, monitor)

function codegen(::Type{Val{:copy}}, nstate::BasicDiscMuvParameterNState, monitor::Vector{Bool})
  body = []
  fnames = fieldnames(BasicDiscMuvParameterNState)
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted

  if monitor[1]
    f = fnames[1]
    push!(
      body,
      :(getfield(_nstate, $(QuoteNode(f)))[1+(_i-1)*_state.size:_i*_state.size] = getfield(_state, $(QuoteNode(f))))
    )
  end

  for j in 2:4
    if monitor[j]
      f = fnames[j]
      push!(body, :(getfield(_nstate, $(QuoteNode(f)))[_i] = getfield(_state, $(QuoteNode(f)))))
    end
  end

  if !isempty(nstate.diagnosticvalues)
    push!(body, :(_nstate.diagnosticvalues[:, _i] = _state.diagnosticvalues))
  end

  @gensym _copy

  quote
    function $_copy(_nstate::BasicDiscMuvParameterNState, _state::BasicDiscMuvParameterState, _i::Integer)
      $(body...)
    end
  end
end

Base.eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscMuvParameterNState{NI, NR}}) = (NI, NR)
Base.eltype{NI<:Integer, NR<:Real}(::BasicDiscMuvParameterNState{NI, NR}) = (NI, NR)

Base.(:(==)){S<:BasicDiscMuvParameterNState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:8]])

Base.isequal{S<:BasicDiscMuvParameterNState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:8]])

function Base.show{NI<:Integer, NR<:Real}(io::IO, nstate::BasicDiscMuvParameterNState{NI, NR})
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
