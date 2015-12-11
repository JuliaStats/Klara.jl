abstract ParameterUpdate <: VariableUpdate

for fname in (
  :LogLikelihood,
  :LogPrior,
  :LogTarget,
  :GradLogLikelihood,
  :GradLogPrior,
  :GradLogTarget,
  :TensorLogLikelihood,
  :TensorLogPrior,
  :TensorLogTarget,
  :DTensorLogLikelihood,
  :DTensorLogPrior,
  :DTensorLogTarget
)
  eval(:(
    immutable $fname{S<:VariableState} <: ParameterUpdate
      states::Vector{S}
    end
  ))

  eval(Expr(:function, Expr(:call, fname, :(nstates::Int)), Expr(:call, fname, :(Array(VariableState, nstates)))))
end
