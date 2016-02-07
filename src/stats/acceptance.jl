function acceptance(v::AbstractArray{Bool}, accept::Bool=true)
  nv = length(v)
  result = (accept ? 100*sum(v)/nv : 100*(nv-sum(v))/nv)
  return result
end

acceptance(v::AbstractArray, accept::Bool=true) = acceptance(convert(AbstractArray{Bool}, v), accept)

acceptance(v::AbstractArray, region, accept::Bool=true) = mapslices(x -> acceptance(x, accept), v, region)

acceptance(s::ParameterNState; key::Symbol=:accept, accept::Bool=true) =
  acceptance(s.diagnosticvalues[findfirst(s.diagnostickeys, key), :], accept)
