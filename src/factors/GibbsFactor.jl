type GibbsFactor <: Factor
  cliques::Vector{Vector{Symbol}}
  logpotentials::FunctionVector
  assignments::Vector{Pair{Symbol, Vector{Symbol}}}
  transforms::FunctionVector
  variables::Vector{Symbol}
  variabletypes::Vector{DataType}
  support::Vector{Union{RealPair, RealPairVector}}
  reparametrize::Vector{Symbol} # :none or :log
  ofvariable::Dict{Symbol, Integer}
end

num_cliques(f::GibbsFactor) = length(f.cliques)

num_transforms(f::GibbsFactor) = length(f.assignments)

num_vertices(f::GibbsFactor) = length(f.variables)

GibbsFactor(
  cliques::Vector{Vector{Symbol}},
  logpotentials::FunctionVector,
  assignments::Vector{Pair{Symbol, Vector{Symbol}}},
  transforms::FunctionVector,
  variables::Vector{Symbol},
  variabletypes::Vector{DataType},
  support::Vector{Union{RealPair, RealPairVector}},
  reparametrize::Vector{Symbol},
  n::Integer=length(variables)
) =
  GibbsFactor(
    cliques,
    logpotentials,
    assignments,
    transforms,
    variables,
    variabletypes,
    support,
    reparametrize,
    Dict(zip(variables, 1:n))
  )

GibbsFactor(
  cliques::Vector{Vector{Symbol}},
  logpotentials::FunctionVector;
  assignments::Vector{Pair{Symbol, Vector{Symbol}}}=Pair{Symbol, Vector{Symbol}}[],
  transforms::FunctionVector=Function[],
  variables::Vector{Symbol}=
    isempty(assignments) ? unique(vcat(cliques...)) : unique(vcat(cliques..., [a.first for a in assignments])),
  variabletypes::Vector{DataType}=DataType[],
  support::Vector{Union{RealPair, RealPairVector}}=Union{RealPair, RealPairVector}[],
  reparametrize::Vector{Symbol}=Symbol[],
  n::Integer=length(variables)
) =
  GibbsFactor(cliques, logpotentials, assignments, transforms, variables, variabletypes, support, reparametrize, n)

function GibbsFactor(
  cliques::Vector{Vector{Symbol}},
  logpotentials::FunctionVector,
  variables::Vector{Symbol},
  variabletypes::Dict{Symbol, DataType};
  assignments::Vector{Pair{Symbol, Vector{Symbol}}}=Pair{Symbol, Vector{Symbol}}[],
  transforms::FunctionVector=Function[],
  support::Dict{Symbol, Union{RealPair, RealPairVector}}=Dict{Symbol, Union{RealPair, RealPairVector}}(),
  reparametrize::Dict{Symbol, Symbol}=Dict{Symbol, Symbol}(),
  n::Integer=length(variables)
)
  local vtypes::Vector{DataType} = Array{DataType}(n)
  local vsupport::Vector{Union{RealPair, RealPairVector}} = Array{Union{RealPair, RealPairVector}}(n)
  local vreparametrize::Vector{Symbol} = Array{Symbol}(n)

  for (v, i) in zip(variables, 1:n)
    vtypes[i] = variabletypes[v]
    vsupport[i] = get(support, v, Pair(-Inf, Inf))
    vreparametrize[i] = get(reparametrize, v, :none)
  end

  println("vtypes = $vtypes")
  GibbsFactor(cliques, logpotentials, assignments, transforms, variables, vtypes, vsupport, vreparametrize, n)
end

GibbsFactor(
  cliques::Vector{Vector{Symbol}},
  logpotentials::FunctionVector,
  variables::Vector{Symbol},
  variabletypes::Dict;
  assignments::Vector{Pair{Symbol, Vector{Symbol}}}=Pair{Symbol, Vector{Symbol}}[],
  transforms::FunctionVector=Function[],
  support::Dict=Dict{Symbol, Union{RealPair, RealPairVector}}(),
  reparametrize::Dict=Dict{Symbol, Symbol}(),
  n::Integer=length(variables)
) =
  GibbsFactor(
    cliques,
    logpotentials,
    variables,
    convert(Dict{Symbol, DataType}, variabletypes),
    assignments=assignments,
    transforms=transforms,
    support=convert(Dict{Symbol, Union{RealPair, RealPairVector}}, support),
    reparametrize=convert(Dict{Symbol, Symbol}, reparametrize),
    n=n
  )

function GibbsFactor(
  cliques::Vector{Vector{Symbol}},
  logpotentials::FunctionVector,
  variabletypes::Dict;
  assignments::Vector{Pair{Symbol, Vector{Symbol}}}=Pair{Symbol, Vector{Symbol}}[],
  transforms::FunctionVector=Function[],
  support::Dict=Dict{Symbol, Union{RealPair, RealPairVector}}(),
  reparametrize::Dict=Dict{Symbol, Symbol}(),
  n::Integer=length(variabletypes)
)
  local variables::Vector{Symbol} = Array{Symbol}(n)
  local vtypes::Vector{DataType} = Array{DataType}(n)
  local vsupport::Vector{Union{RealPair, RealPairVector}} = Array{Union{RealPair, RealPairVector}}(n)
  local vreparametrize::Vector{Symbol} = Array{Symbol}(n)
  local i::Int64 = 1

  for (k, t) in variabletypes
    variables[i] = k
    vtypes[i] = t
    vsupport[i] = get(support, k, Pair(-Inf, Inf))
    vreparametrize[i] = get(reparametrize, k, :none)
    i += 1
  end

  return GibbsFactor(cliques, logpotentials, assignments, transforms, variables, vtypes, vsupport, vreparametrize, n)
end

function generate_logtarget(
  f::GibbsFactor, i::Integer, stream::IOStream, ::Type{Univariate}, nc::Integer=num_cliques(f), nv::Integer=num_vertices(f)
)
  local body = ["_state.logtarget = 0."]
  local lpargs::Vector{String}
  local lptargs::Vector{Int8} = fill(Int8(0), nv)

  for j in 1:nc
    if in(f.variables[i], f.cliques[j])
      lpargs = []

      for v in f.cliques[j]
        if v == f.variables[i]
          if f.support[f.ofvariable[v]] == Pair(0., Inf) && f.reparametrize[f.ofvariable[v]] == :log
            push!(lpargs, "exp(_state.value)")
            lptargs[f.ofvariable[v]] = Int8(1)
          else
            push!(lpargs, "_state.value")
          end
        else
          if f.support[f.ofvariable[v]] == Pair(0., Inf) && f.reparametrize[f.ofvariable[v]] == :log
            push!(lpargs, "exp(_states[$(f.ofvariable[v])].value)")
            lptargs[f.ofvariable[v]] = Int8(2)
          else
            push!(lpargs, "_states[$(f.ofvariable[v])].value")
          end
        end
      end

      push!(body, "_state.logtarget += f.logpotentials[$j]($(join(lpargs, ", ")))")
    end
  end

  for k in 1:nv
    if lptargs[k] == 1
      push!(body, "_state.logtarget += _state.value")
    elseif lptargs[k] == 2
      push!(body, "_state.logtarget += _states[$k].value")
    end
  end

  write(
    stream,
    string(
      "function logtarget_",
      f.variables[1],
      "(_state::",
      default_state_type(f.variabletypes[i]),
      "_states::VariableStateVector)\n")
  )

  for ln in body
    write(stream, "  $(ln)\n")
  end

  write(stream, "end\n")
end

function generate_logtarget(
  f::GibbsFactor,
  i::Integer,
  filename::AbstractString,
  vform::Type{Univariate},
  nc::Integer=num_cliques(f),
  nv::Integer=num_vertices(f)
)
  stream = open(filename, "w")
  generate_logtarget(f, i, stream, vform, nc, nv)
  close(stream)
end

function codegen_logtarget(
  f::GibbsFactor, i::Integer, ::Type{Univariate}, nc::Integer=num_cliques(f), nv::Integer=num_vertices(f)
)
  local body = [:(_state.logtarget = 0.)]
  local lpargs::Vector{Expr}
  local lptargs::Vector{Int8} = fill(Int8(0), nv)

  for j in 1:nc
    if in(f.variables[i], f.cliques[j])
      lpargs = []

      for v in f.cliques[j]
        if v == f.variables[i]
          if f.support[f.ofvariable[v]] == Pair(0., Inf) && f.reparametrize[f.ofvariable[v]] == :log
            push!(lpargs, :(exp(_state.value)))
            lptargs[f.ofvariable[v]] = Int8(1)
          else
            push!(lpargs, :(_state.value))
          end
        else
          if f.support[f.ofvariable[v]] == Pair(0., Inf) && f.reparametrize[f.ofvariable[v]] == :log
            push!(lpargs, :(exp(_states[$(f.ofvariable[v])].value)))
            lptargs[f.ofvariable[v]] = Int8(2)
          else
            push!(lpargs, :(_states[$(f.ofvariable[v])].value))
          end
        end
      end

      push!(body, :(_state.logtarget += f.logpotentials[$j]($(lpargs...))))
    end
  end

  for k in 1:nv
    if lptargs[k] == 1
      push!(body, :(_state.logtarget += _state.value))
    elseif lptargs[k] == 2
      push!(body, :(_state.logtarget += _states[$k].value))
    end
  end

  @gensym logtarget

  quote
    function $logtarget(_state::$(default_state_type(f.variabletypes[i])), _states::VariableStateVector)
      $(body...)
    end
  end
end

function codegen_logtarget(
  f::GibbsFactor, i::Integer, ::Type{Multivariate}, nc::Integer=num_cliques(f), nv::Integer=num_vertices(f)
)
  local body = [:(_state.logtarget = 0.)]
  local lpargs::Vector{Expr}

  for j in 1:nc
    if in(f.variables[i], f.cliques[j])
      lpargs = []

      for v in f.cliques[j]
        if v == f.variables[i]
          push!(lpargs, :(_state.value))
        else
          push!(lpargs, :(_states[$(f.ofvariable[v])].value))
        end
      end

      push!(body, :(_state.logtarget += f.logpotentials[$j]($(lpargs...))))
    end
  end

  @gensym logtarget

  quote
    function $logtarget(_state::$(default_state_type(f.variabletypes[i])), _states::VariableStateVector)
      $(body...)
    end
  end
end

function codegen_transform(f::GibbsFactor, i::Integer, nt::Integer=num_transforms(f))
  local transformations::Vector{Symbol} = Symbol[a.first for a in f.assignments]

  local k::Integer = findfirst(transformations, f.variables[i])
  @assert k != 0 "Transformation function for variable $(f.variables[i]) has not been provided"

  local lpargs::Vector{Expr} = Expr[]

  for v in f.assignments[k].second
    push!(lpargs, :(_states[$(f.ofvariable[v])].value))
  end

  @gensym transform

  quote
    function $transform(_states::VariableStateVector)
      f.transforms[$k]($(lpargs...))
    end
  end
end

function generate_variables(f::GibbsFactor, nc::Integer=num_cliques(f), nv::Integer=num_vertices(f))
  local variables::VariableVector = Array{Variable}(nv)

  for i in 1:nv
    if issubtype(f.variabletypes[i], Parameter)
      variables[i] = f.variabletypes[i](
        f.variables[i], signature=:low, logtarget=eval(codegen_logtarget(f, i, variate_form(f.variabletypes[i]), nc, nv))
      )
    else
      variables[i] = f.variabletypes[i](f.variables[i])
    end
  end

  return variables
end
