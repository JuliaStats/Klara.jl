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

function codegen_forward_autodiff_uptofunction(method::Symbol, f::Function, chunksize::Int=0)
  adfunction = forward_autodiff_function(method, f, true, chunksize)

  if method == :derivative
    @gensym forward_autodiff_uptofunction
    quote
      function $forward_autodiff_uptofunction(_x::Real)
        result, allresults = $(adfunction)(_x)
        return ForwardDiff.value(allresults), result
      end
    end
  elseif method == :gradient
    @gensym forward_autodiff_uptofunction
    quote
      function $forward_autodiff_uptofunction(_x::Vector)
        result, allresults = $(adfunction)(_x)
        return ForwardDiff.value(allresults), result
      end
    end
  elseif method == :hessian
    @gensym forward_autodiff_uptofunction
    quote
      function $forward_autodiff_uptofunction(_x::Vector)
        result, allresults = $(adfunction)(_x)
        return ForwardDiff.value(allresults), ForwardDiff.gradient(allresults), -result
      end
    end
  else
    error("No support for generation of forward mode autodiff uptofunction for $method")
  end
end
