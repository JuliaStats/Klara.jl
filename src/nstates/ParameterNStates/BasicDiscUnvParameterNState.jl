type BasicDiscUnvParameterNState{NI<:Integer, NR<:Real} <: ParameterNState{Discrete, Univariate}
  value::Vector{NI}
  loglikelihood::Vector{NR}
  logprior::Vector{NR}
  logtarget::Vector{NR}
  diagnosticvalues::Matrix
  n::Integer
  diagnostickeys::Vector{Symbol}
  copy::Function

  function BasicDiscUnvParameterNState(
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 3)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{NI}=Int64,
    ::Type{NR}=Float64,
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  )
    instance = new()

    t = [NI; fill(NR, 3)]

    l = Array(Integer, 4)
    for i in 1:4
      l[i] = (monitor[i] == false ? zero(Integer) : n)
    end

    fnames = fieldnames(BasicDiscUnvParameterNState)
    for i in 1:4
      setfield!(instance, fnames[i], Array(t[i], l[i]))
    end

    instance.diagnosticvalues = diagnosticvalues

    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance.copy = eval(codegen(:copy, instance, monitor))

    instance
  end
end

BasicDiscUnvParameterNState{NI<:Integer, NR<:Real}(
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 3)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) =
  BasicDiscUnvParameterNState{NI, NR}(n, monitor, diagnostickeys, NI, NR, diagnosticvalues)

function BasicDiscUnvParameterNState{NI<:Integer, NR<:Real}(
  n::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{NI}=Int64,
  ::Type{NR}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
)
  fnames = fieldnames(BasicDiscUnvParameterNState)
  BasicDiscUnvParameterNState(
    n, [fnames[i] in monitor ? true : false for i in 1:4], diagnostickeys, NI, NR, diagnosticvalues
  )
end

typealias DiscUnvMarkovChain BasicDiscUnvParameterNState

codegen(f::Symbol, nstate::BasicDiscUnvParameterNState, monitor::Vector{Bool}) = codegen(Val{f}, nstate, monitor)

function codegen(::Type{Val{:copy}}, nstate::BasicDiscUnvParameterNState, monitor::Vector{Bool})
  body = []
  fnames = fieldnames(BasicDiscUnvParameterNState)
  local f::Symbol # f must be local to avoid compiler errors. Alternatively, this variable declaration can be omitted

  for j in 1:4
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
    function $_copy(_nstate::BasicDiscUnvParameterNState, _state::BasicDiscUnvParameterState, _i::Integer)
      $(body...)
    end
  end
end

Base.eltype{NI<:Integer, NR<:Real}(::Type{BasicDiscUnvParameterNState{NI, NR}}) = (NI, NR)
Base.eltype{NI<:Integer, NR<:Real}(::BasicDiscUnvParameterNState{NI, NR}) = (NI, NR)

Base.(:(==)){S<:BasicDiscUnvParameterNState}(z::S, w::S) =
  reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(S)[1:7]])

Base.isequal{S<:BasicDiscUnvParameterNState}(z::S, w::S) =
  reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(S)[1:7]])

function Base.show{NI<:Integer, NR<:Real}(io::IO, nstate::BasicDiscUnvParameterNState{NI, NR})
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
