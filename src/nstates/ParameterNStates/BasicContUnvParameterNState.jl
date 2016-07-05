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
  n::Integer
  diagnostickeys::Vector{Symbol}
  copy::Function

  function BasicContUnvParameterNState(
    n::Integer,
    monitor::Vector{Bool}=[true; fill(false, 12)],
    diagnostickeys::Vector{Symbol}=Symbol[],
    ::Type{N}=Float64,
    diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
  )
    instance = new()

    l = Array(Integer, 13)
    for i in 1:13
      l[i] = (monitor[i] == false ? zero(Integer) : n)
    end

    fnames = fieldnames(BasicContUnvParameterNState)
    for i in 1:13
      setfield!(instance, fnames[i], Array(N, l[i]))
    end

    instance.diagnosticvalues = diagnosticvalues

    instance.n = n
    instance.diagnostickeys = diagnostickeys

    instance.copy = eval(codegen(:copy, instance, monitor))

    instance
  end
end

BasicContUnvParameterNState{N<:Real}(
  n::Integer,
  monitor::Vector{Bool}=[true; fill(false, 12)],
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
) =
  BasicContUnvParameterNState{N}(n, monitor, diagnostickeys, N, diagnosticvalues)

function BasicContUnvParameterNState{N<:Real}(
  n::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diagnosticvalues::Matrix=Array(Any, length(diagnostickeys), isempty(diagnostickeys) ? 0 : n)
)
  fnames = fieldnames(BasicContUnvParameterNState)
  BasicContUnvParameterNState(n, [fnames[i] in monitor ? true : false for i in 1:13], diagnostickeys, N, diagnosticvalues)
end

typealias ContUnvMarkovChain BasicContUnvParameterNState

codegen(f::Symbol, nstate::BasicContUnvParameterNState, monitor::Vector{Bool}) = codegen(Val{f}, nstate, monitor)

function codegen(::Type{Val{:copy}}, nstate::BasicContUnvParameterNState, monitor::Vector{Bool})
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

  @gensym _copy

  quote
    function $_copy(_nstate::BasicContUnvParameterNState, _state::BasicContUnvParameterState, _i::Integer)
      $(body...)
    end
  end
end

Base.eltype{N<:Real}(::Type{BasicContUnvParameterNState{N}}) = N
Base.eltype{N<:Real}(::BasicContUnvParameterNState{N}) = N

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
