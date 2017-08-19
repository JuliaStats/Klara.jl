function set_autodiff_function(::Type{Val{:forward}}, ::Type{Val{:derivative}})
  function (result::DiffBase.DiffResult, f::Function, x::Real)
    ForwardDiff.derivative!(result, f, x)
    return DiffBase.derivative(result)
  end
end

function set_autodiff_function(::Type{Val{:forward}}, ::Type{Val{:gradient}})
  function (result::DiffBase.DiffResult, f::Function, x::Vector, cfg::ForwardDiff.GradientConfig)
    ForwardDiff.gradient!(result, f, x, cfg)
    return DiffBase.gradient(result)
  end
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

function codegen_autodiff_uptofunction(::Type{Val{:forward}}, ::Type{Val{:derivative}})
  body = []

  push!(body, :(getfield(ForwardDiff, :derivative!)(_result, _f, _x)))

  push!(body, :(return DiffBase.value(_result), DiffBase.derivative(_result)))

  @gensym autodiff_uptofunction
  quote
    function $autodiff_uptofunction(_result::DiffBase.DiffResult, _f::Function, _x::Real)
      $(body...)
    end
  end
end

function codegen_autodiff_uptofunction(::Type{Val{:forward}}, ::Type{Val{:gradient}})
  body = []

  push!(body, :(getfield(ForwardDiff, :gradient!)(_result, _f, _x, _cfg)))

  push!(body, :(return DiffBase.value(_result), DiffBase.gradient(_result)))

  @gensym autodiff_uptofunction
  quote
    function $autodiff_uptofunction(_result::DiffBase.DiffResult, _f::Function, _x::Vector, _cfg::ForwardDiff.GradientConfig)
      $(body...)
    end
  end
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
