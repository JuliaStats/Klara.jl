function forward_autodiff_derivative(result::DiffBase.DiffResult, f::Function, x::Real)
  ForwardDiff.derivative!(result, f, x)
  return DiffBase.derivative(result)
end

function forward_autodiff_gradient(result::DiffBase.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.GradientConfig)
  ForwardDiff.gradient!(result, f, x, cfg)
  return DiffBase.gradient(result)
end

function forward_autodiff_negative_hessian(
  result::DiffBase.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.HessianConfig
)
  ForwardDiff.hessian!(result, f, x, cfg)
  return -DiffBase.hessian(result)
end

function forward_autodiff_upto_derivative(result::DiffBase.DiffResult, f::Function, x::Real)
  ForwardDiff.derivative!(result, f, x)
  return DiffBase.value(result), DiffBase.derivative(result)
end

function forward_autodiff_upto_gradient(result::DiffBase.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.GradientConfig)
  ForwardDiff.gradient!(result, f, x, cfg)
  return DiffBase.value(result), DiffBase.gradient(result)
end

function forward_autodiff_upto_negative_hessian(
  result::DiffBase.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.HessianConfig
)
  ForwardDiff.hessian!(result, f, x, cfg)
  return DiffBase.value(result), DiffBase.gradient(result), -DiffBase.hessian(result)
end
