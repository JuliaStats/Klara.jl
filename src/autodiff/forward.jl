function codegen_forward_autodiff_function(::Type{Val{:derivative}}, f::Function)
  @gensym forward_autodiff_function
  quote
    function $forward_autodiff_function(_x::Real)
      getfield(ForwardDiff, :derivative)($f, _x)
    end
  end
end

function codegen_forward_autodiff_function(::Type{Val{:gradient}}, f::Function, chunksize::Integer=0)
  body =
    if chunksize == 0
      :(getfield(ForwardDiff, :gradient)($f, _x))
    else
      :(getfield(ForwardDiff, :gradient)($f, _x, Chunk{$chunksize}()))
    end

  @gensym forward_autodiff_function
  quote
    function $forward_autodiff_function(_x::Vector)
      $body
    end
  end
end

codegen_forward_autodiff_target(method::Symbol, f::Function, chunksize::Integer=0) =
  codegen_forward_autodiff_target(Val{method}, f, chunksize)

function codegen_forward_autodiff_target(::Type{Val{:hessian}}, f::Function, chunksize::Integer=0)
  body =
    if chunksize == 0
      :(-getfield(ForwardDiff, :hessian)($f, _x))
    else
      :(-getfield(ForwardDiff, :hessian)($f, _x, Chunk{$chunksize}()))
    end

  @gensym forward_autodiff_target
  quote
    function $forward_autodiff_target(_x::Vector)
      $body
    end
  end
end

function codegen_forward_autodiff_uptofunction(::Type{Val{:derivative}}, f::Function)
  @gensym forward_autodiff_uptofunction
  quote
    function $forward_autodiff_uptofunction(_x::Real)
      result = DiffBase.DiffResult(_x, _x)
      getfield(ForwardDiff, :derivative!)(result, $f, _x)
      return DiffBase.value(result), DiffBase.derivative(result)
    end
  end
end

function codegen_forward_autodiff_uptofunction(::Type{Val{:gradient}}, f::Function, chunksize::Integer=0)
  adfcall =
    if chunksize == 0
      :(getfield(ForwardDiff, :gradient!)(result, $f, _x))
    else
      :(getfield(ForwardDiff, :gradient!)(result, $f, _x, Chunk{$chunksize}()))
    end

  @gensym forward_autodiff_uptofunction
  quote
    function $forward_autodiff_uptofunction(_x::Vector)
      result = DiffBase.GradientResult(_x)
      $adfcall
      return DiffBase.value(result), DiffBase.gradient(result)
    end
  end
end

codegen_forward_autodiff_uptotarget(method::Symbol, f::Function, chunksize::Integer=0) =
  codegen_forward_autodiff_uptotarget(Val{method}, f, chunksize)

function codegen_forward_autodiff_uptotarget(::Type{Val{:hessian}}, f::Function, chunksize::Integer=0)
  adfcall =
    if chunksize == 0
      :(getfield(ForwardDiff, :hessian!)(result, $f, _x))
    else
      :(getfield(ForwardDiff, :hessian!)(result, $f, _x, Chunk{$chunksize}()))
    end

  @gensym forward_autodiff_uptotarget
  quote
    function $forward_autodiff_uptotarget(_x::Vector)
      result = DiffBase.HessianResult(_x)
      $adfcall
      return DiffBase.value(result), DiffBase.gradient(result), -DiffBase.hessian(result)
    end
  end
end
