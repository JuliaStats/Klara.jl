function generate_logtarget(
  f::GibbsFactor, i::Integer, stream::IOStream, ::Type{Klara.Univariate}, nc::Integer=num_cliques(f), nv::Integer=num_vertices(f)
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
      Klara.default_state_type(f.variabletypes[i]),
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
  vform::Type{Klara.Univariate},
  nc::Integer=num_cliques(f),
  nv::Integer=num_vertices(f)
)
  stream = open(filename, "w")
  generate_logtarget(f, i, stream, vform, nc, nv)
  close(stream)
end

generate_logtarget(f, 1, "test.jl", Klara.variate_form(BasicContUnvParameter), 4)
