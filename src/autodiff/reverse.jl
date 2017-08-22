function reverse_autodiff_gradient(
  result::DiffBase.DiffResult,
  f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.gradient!(result, f, x)
  return DiffBase.gradient(result)
end

function codegen_autodiff_target(::Type{Val{:reverse}}, ::Type{Val{:hessian}})
  body = []

  push!(body, :(getfield(ReverseDiff, :hessian!)(_result, _f, _x)))

  push!(body, :(return -DiffBase.hessian(_result)))

  @gensym autodiff_target
  quote
    function $autodiff_target(
      _result::DiffBase.DiffResult, _f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape}, _x::Vector
    )
      $(body...)
    end
  end
end

function reverse_autodiff_upto_gradient(
  result::DiffBase.DiffResult,
  f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape},
  x::Vector
)
  ReverseDiff.gradient!(result, f, x)
  return DiffBase.value(result), DiffBase.gradient(result)
end

function codegen_autodiff_uptotarget(::Type{Val{:reverse}}, ::Type{Val{:hessian}})
  body = []

  push!(body, :(getfield(ReverseDiff, :hessian!)(_result, _f, _x)))

  push!(body, :(return DiffBase.value(_result), DiffBase.gradient(_result), -DiffBase.hessian(_result)))

  @gensym autodiff_uptotarget
  quote
    function $autodiff_uptotarget(
      _result::DiffBase.DiffResult, _f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape}, _x::Vector
    )
      $(body...)
    end
  end
end
