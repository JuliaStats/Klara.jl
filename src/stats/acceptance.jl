acceptance(v::AbstractArray{Bool}) = mean(v)

function acceptance(v::AbstractArray)
  accepted = 1
  n = length(v)

  for i in 2:n
    if v[i] != v[i-1]
      accepted += 1
    end
  end

  return accepted/n
end

acceptance(v::AbstractArray, region) = mapslices(acceptance, v, region)

function acceptance(s::UnivariateParameterNState; key::Symbol=:accept, diagnostics::Bool=true)
  if diagnostics
    return acceptance(convert(Vector{Bool}, s.diagnosticvalues[findfirst(s.diagnostickeys, key), :]))
  else
    return acceptance(s.value)
  end
end

acceptance(s::VariableNState{Univariate}) = acceptance(s.value)

function acceptance(s::MultivariateParameterNState; key::Symbol=:accept, diagnostics::Bool=true)
  if diagnostics
    return acceptance(convert(Vector{Bool}, s.diagnosticvalues[findfirst(s.diagnostickeys, key), :]))
  else
    return acceptance(Any[s.value[:, i] for i in 1:s.n])
  end
end

acceptance(s::VariableNState{Multivariate}) = acceptance(Any[s.value[:, i] for i in 1:s.n])
