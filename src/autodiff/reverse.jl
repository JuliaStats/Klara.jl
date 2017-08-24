function reverse_autodiff_gradient(
  result::DiffBase.DiffResult,
  f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.gradient!(result, f, x)
  return DiffBase.gradient(result)
end

function reverse_autodiff_negative_hessian(
  result::DiffBase.DiffResult,
  f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.hessian!(result, f, x)
  return -DiffBase.hessian(result)
end

function reverse_autodiff_upto_gradient(
  result::DiffBase.DiffResult,
  f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.gradient!(result, f, x)
  return DiffBase.value(result), DiffBase.gradient(result)
end

function reverse_autodiff_upto_negative_hessian(
  result::DiffBase.DiffResult,
  f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.hessian!(result, f, x)
  return DiffBase.value(result), DiffBase.gradient(result), -DiffBase.hessian(result)
end
