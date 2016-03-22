function forward_autodiff_function(method::Symbol, f::Function, allresults::Bool=false, chunksize::Int=0)
  args = Any[f]
  if allresults
    push!(args, ForwardDiff.AllResults)
  end

  if chunksize == 0
    getfield(ForwardDiff, method)(args...)
  else
    getfield(ForwardDiff, method)(args..., chunk_size=chunksize)
  end
end

codegen_forward_autodiff_target(method::Symbol, f::Function, chunksize::Int=0) =
  codegen_forward_autodiff_target(Val{method}, f, chunksize)

function codegen_forward_autodiff_target(::Type{Val{:hessian}}, f::Function, chunksize::Int=0)
  adfunction = forward_autodiff_function(:hessian, f, false, chunksize)

  @gensym forward_autodiff_target
  quote
    function $forward_autodiff_target(_x::Vector)
      -$(adfunction)(_x)
    end
  end
end

codegen_forward_autodiff_uptofunction(method::Symbol, f::Function, chunksize::Int=0) =
  codegen_forward_autodiff_uptofunction(Val{method}, f, chunksize)

function codegen_forward_autodiff_uptofunction(::Type{Val{:derivative}}, f::Function, chunksize::Int=0)
  adfunction = forward_autodiff_function(:derivative, f, true, chunksize)

  @gensym forward_autodiff_uptofunction
  quote
    function $forward_autodiff_uptofunction(_x::Real)
      result, allresults = $(adfunction)(_x)
      return ForwardDiff.value(allresults), result
    end
  end
end

function codegen_forward_autodiff_uptofunction(::Type{Val{:gradient}}, f::Function, chunksize::Int=0)
  adfunction = forward_autodiff_function(:gradient, f, true, chunksize)

  @gensym forward_autodiff_uptofunction
  quote
    function $forward_autodiff_uptofunction(_x::Vector)
      result, allresults = $(adfunction)(_x)
      return ForwardDiff.value(allresults), result
    end
  end
end

codegen_forward_autodiff_uptotarget(method::Symbol, f::Function, chunksize::Int=0) =
  codegen_forward_autodiff_uptotarget(Val{method}, f, chunksize)

function codegen_forward_autodiff_uptotarget(::Type{Val{:hessian}}, f::Function, chunksize::Int=0)
  adfunction = forward_autodiff_function(:hessian, f, true, chunksize)

  @gensym forward_autodiff_uptotarget
  quote
    function $forward_autodiff_uptotarget(_x::Vector)
      result, allresults = $(adfunction)(_x)
      return ForwardDiff.value(allresults), ForwardDiff.gradient(allresults), -result
    end
  end
end
