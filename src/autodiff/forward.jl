function forward_autodiff_derivative(result::DiffBase.DiffResult, f::Function, x::Real)
  ForwardDiff.derivative!(result, f, x)
  return DiffBase.derivative(result)
end

function forward_autodiff_gradient(result::DiffBase.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.GradientConfig)
  ForwardDiff.gradient!(result, f, x, cfg)
  return DiffBase.gradient(result)
end

function codegen_autodiff_target(::Type{Val{:forward}}, ::Type{Val{:hessian}})
  body = []

  push!(body, :(getfield(ForwardDiff, :hessian!)(_result, _f, _x, _cfg)))

  push!(body, :(return -DiffBase.hessian(_result)))

  @gensym autodiff_target
  quote
    function $autodiff_target(_result::DiffBase.DiffResult, _f::Function, _x::Vector, _cfg::ForwardDiff.HessianConfig)
      $(body...)
    end
  end
end

function forward_autodiff_upto_derivative(result::DiffBase.DiffResult, f::Function, x::Real)
  ForwardDiff.derivative!(result, f, x)
  return DiffBase.value(result), DiffBase.derivative(result)
end

function forward_autodiff_upto_gradient(result::DiffBase.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.GradientConfig)
  ForwardDiff.gradient!(result, f, x, cfg)
  return DiffBase.value(result), DiffBase.gradient(result)
end

function codegen_autodiff_uptotarget(::Type{Val{:forward}}, ::Type{Val{:hessian}})
  body = []

  push!(body, :(getfield(ForwardDiff, :hessian!)(_result, _f, _x, _cfg)))

  push!(body, :(return DiffBase.value(_result), DiffBase.gradient(_result), -DiffBase.hessian(_result)))

  @gensym autodiff_uptotarget
  quote
    function $autodiff_uptotarget(_result::DiffBase.DiffResult, _f::Function, _x::Vector, _cfg::ForwardDiff.HessianConfig)
      $(body...)
    end
  end
end
