function codegen_autodiff_function(::Type{Val{:reverse}}, ::Type{Val{:gradient}})
  body = []

  push!(body, :(getfield(ReverseDiff, :gradient!)(_result, _f, _x)))
  # push!(body, :(getfield(ReverseDiff, :gradient!)(_result, _f, _x, _cfg)))

  push!(body, :(return DiffBase.gradient(_result)))

  @gensym autodiff_function
  quote
    function $autodiff_function(
      _result::DiffBase.DiffResult, _f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape}, _x::Vector
    )
    # function $autodiff_function(
    #   _result::DiffBase.DiffResult,
    #   _f:Union{ReverseDiff.GradientTape,
    #   ReverseDiff.CompiledTape},
    #   _x::Vector,
    #   _cfg::ReverseDiff.GradientConfig
    # )
      $(body...)
    end
  end
end

function codegen_autodiff_target(::Type{Val{:reverse}}, ::Type{Val{:hessian}})
  body = []

  push!(body, :(getfield(ReverseDiff, :hessian!)(_result, _f, _x)))
  # push!(body, :(getfield(ReverseDiff, :hessian!)(_result, _f, _x, _cfg)))

  push!(body, :(return -DiffBase.hessian(_result)))

  @gensym autodiff_target
  quote
    function $autodiff_target(
      _result::DiffBase.DiffResult, _f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape}, _x::Vector
    )
    # function $autodiff_target(
    #   _result::DiffBase.DiffResult,
    #   _f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape},
    #   _x::Vector,
    #   _cfg::ReverseDiff.HessianConfig
    # )
      $(body...)
    end
  end
end

function codegen_autodiff_uptofunction(::Type{Val{:reverse}}, ::Type{Val{:gradient}})
  body = []

  push!(body, :(getfield(ReverseDiff, :gradient!)(_result, _f, _x)))
  # push!(body, :(getfield(ReverseDiff, :gradient!)(_result, _f, _x, _cfg)))

  push!(body, :(return DiffBase.value(_result), DiffBase.gradient(_result)))

  @gensym autodiff_uptofunction
  quote
    function $autodiff_uptofunction(
      _result::DiffBase.DiffResult, _f::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape}, _x::Vector
    )
    # function $autodiff_uptofunction(
    #   _result::DiffBase.DiffResult,
    #   _f:Union{ReverseDiff.GradientTape,
    #   ReverseDiff.CompiledTape},
    #   _x::Vector,
    #   _cfg::ReverseDiff.GradientConfig
    # )
      $(body...)
    end
  end
end

function codegen_autodiff_uptotarget(::Type{Val{:reverse}}, ::Type{Val{:hessian}})
  body = []

  push!(body, :(getfield(ReverseDiff, :hessian!)(_result, _f, _x)))
  # push!(body, :(getfield(ReverseDiff, :hessian!)(_result, _f, _x, _cfg)))

  push!(body, :(return DiffBase.value(_result), DiffBase.gradient(_result), -DiffBase.hessian(_result)))

  @gensym autodiff_uptotarget
  quote
    function $autodiff_uptotarget(
      _result::DiffBase.DiffResult, _f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape}, _x::Vector
    )
    # function $autodiff_uptotarget(
    #   _result::DiffBase.DiffResult,
    #   _f::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape},
    #   _x::Vector,
    #   _cfg::ReverseDiff.HessianConfig
    # )
      $(body...)
    end
  end
end
