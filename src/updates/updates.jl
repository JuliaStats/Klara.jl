abstract VariableUpdate

for fname in (:DataUpdate, :Transform)
  eval(:(
    immutable $fname{S<:VariableState} <: VariableUpdate
      states::Vector{S}
    end
  ))

  eval(Expr(:function, Expr(:call, fname, :(nstates::Int)), Expr(:call, fname, :(Array(VariableState, nstates)))))
end
