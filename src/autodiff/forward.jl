function forward_autodiff_derivative(result::DiffResults.DiffResult, f::Function, x::Real)
  ForwardDiff.derivative!(result, f, x)
  return DiffResults.derivative(result)
end

function forward_autodiff_gradient(result::DiffResults.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.GradientConfig)
  result = ForwardDiff.gradient!(result, f, x, cfg)
  return DiffResults.gradient(result)
end

function forward_autodiff_negative_hessian(
  result::DiffResults.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.HessianConfig
)
  ForwardDiff.hessian!(result, f, x, cfg)
  return -DiffResults.hessian(result)
end

function forward_autodiff_upto_derivative(result::DiffResults.DiffResult, f::Function, x::Real)
  ForwardDiff.derivative!(result, f, x)
  return DiffResults.value(result), DiffResults.derivative(result)
end

function forward_autodiff_upto_gradient(
  result::DiffResults.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.GradientConfig
)
  ForwardDiff.gradient!(result, f, x, cfg)
  return DiffResults.value(result), DiffResults.gradient(result)
end

function forward_autodiff_upto_negative_hessian(
  result::DiffResults.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.HessianConfig
)
  ForwardDiff.hessian!(result, f, x, cfg)
  return DiffResults.value(result), DiffResults.gradient(result), -DiffResults.hessian(result)
end
