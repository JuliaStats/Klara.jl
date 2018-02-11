function reverse_autodiff_gradient(
  result::DiffResults.DiffResult,
  f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.gradient!(result, f, x)
  return DiffResults.gradient(result)
end

function reverse_autodiff_negative_hessian(
  result::DiffResults.DiffResult,
  f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.hessian!(result, f, x)
  return -DiffResults.hessian(result)
end

function reverse_autodiff_upto_gradient(
  result::DiffResults.DiffResult,
  f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.gradient!(result, f, x)
  return DiffResults.value(result), DiffResults.gradient(result)
end

function reverse_autodiff_upto_negative_hessian(
  result::DiffResults.DiffResult,
  f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.hessian!(result, f, x)
  return DiffResults.value(result), DiffResults.gradient(result), -DiffResults.hessian(result)
end
