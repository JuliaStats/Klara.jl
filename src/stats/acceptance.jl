acceptance(v::AbstractArray{Bool}, accept::Bool=true) = (accept ? mean(v) : 1-mean(v))

acceptance(v::AbstractArray, accept::Bool=true) = acceptance(convert(AbstractArray{Bool}, v), accept)

acceptance(v::AbstractArray, region, accept::Bool=true) = mapslices(x -> acceptance(x, accept), v, region)

acceptance(s::ParameterNState; key::Symbol=:accept, accept::Bool=true) =
  acceptance(s.diagnosticvalues[findfirst(s.diagnostickeys, key), :], accept)
