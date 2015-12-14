type VariableFunctor{S<:VariableState}
  f::Function
  exprn::Expr
  states::Vector{S}
end

VariableFunctor{S<:VariableState}(f::Function, exprn::Expr,  states::S) = VariableFunctor{S}(f, exprn, states)

VariableFunctor(f::Function, exprn=:()) = VariableFunctor(f, exprn, VariableState[])
