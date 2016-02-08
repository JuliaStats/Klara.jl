acceptance(v::AbstractArray{Bool}) = mean(v)

function acceptance{N<:Number}(v::AbstractArray{N})
  accepted = 1
  n = length(v)

  for i in 2:n
    if v[i] != v[i-1]
      accepted += 1
    end
  end

  return accepted/n
end

acceptance(v::AbstractArray) = acceptance(convert(AbstractArray{Bool}, v))

acceptance(v::AbstractArray, region) = mapslices(acceptance, v, region)

function acceptance(s::ParameterNState, key::Symbol=:accept)
  i = findfirst(s.diagnostickeys, key)
  i == 0 ? acceptance(s.value) : acceptance(s.diagnosticvalues[i, :])
end

acceptance(s::VariableNState, key::Symbol=:accept) = acceptance(s.value)
