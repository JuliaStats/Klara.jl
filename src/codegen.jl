function codegen_lowlevel_variable_method(
  f::Function,
  r::Vector{Symbol},
  statetype::Union{Symbol, Void},
  nkeys::Int,
  vfarg::Bool
)
  body::Expr
  fargs::Vector
  rvalues::Expr

  if nkeys == 0
    fargs = [:(_state.value)]
  elseif nkeys > 0
    if vfarg
      fargs = [Expr(:ref, :Any, [:(_states[$i].value) for i in 1:nkeys]...)]
    else
      fargs = [:(_state.value), Expr(:ref, :Any, [:(_states[$i].value) for i in 1:nkeys]...)]
    end
  else
    error("nkeys must be non-negative, got $nkeys")
  end

  nr = length(r)
  if nr == 0
    body = :($(f)($(fargs...)))
  elseif nr == 1
    rvalues = Expr(:., :_state, QuoteNode(r[1]))
    body = :($rvalues = $(f)($(fargs...)))
  elseif nr > 1
    rvalues = Expr(:tuple, [Expr(:., :_state, QuoteNode(r[i])) for i in 1:nr]...)
    body = :($rvalues = $(f)($(fargs...)))
  else
    error("Vector of return symbols must have one or more elements")
  end

  statearg = statetype == nothing ? :_state : :(_state::$statetype)
  
  @gensym lowlevel_variable_method

  quote
    function $lowlevel_variable_method($statearg, _states::VariableStateVector)
      $body
    end
  end
end
