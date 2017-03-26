function codegen_lowlevel_variable_method(
  f::Function;
  statetype::Union{Symbol, Void}=nothing,
  states::Bool=true,
  returns::Vector{Symbol}=Symbol[],
  vfarg::Bool=false,
  nkeys::Integer=0,
  diffresult::Union{Symbol, Void}=nothing,
  diffmethod::Union{Symbol, Void}=nothing,
  diffconfig::Union{Symbol, Void}=nothing
)
  local body::Expr
  local fargs::Vector = []
  local rvalues::Expr

  if vfarg
    @assert nkeys > 0 "If vfarg=$vfarg, nkeys must be positive, got $nkeys"
    @assert diffresult == nothing "If vfarg=$vfarg, diffresult must be nothing, got $diffresult"
  else
    @assert nkeys >= 0 "If vfarg=$vfarg, nkeys must be non-negative, got $nkeys"
  end

  if vfarg
    fargs = [Expr(:ref, :Any, [:(_states[$i].value) for i in 1:nkeys]...)]
  else
    if diffresult != nothing
      fargs = [Expr(:., :(_state.diffstate), QuoteNode(diffresult))]
    end

    if diffmethod != nothing
      push!(fargs, Expr(:., :(_state.diffmethods), QuoteNode(diffmethod)))
    end

    push!(fargs, :(_state.value))

    if diffconfig != nothing
      push!(fargs, Expr(:., :(_state.diffstate), QuoteNode(diffconfig)))
    end

    if nkeys > 0
      push!(fargs, Expr(:ref, :Any, [:(_states[$i].value) for i in 1:nkeys]...))
    end
  end

  nr = length(returns)
  if nr == 0
    body = :($(f)($(fargs...)))
  elseif nr == 1
    rvalues = Expr(:., :_state, QuoteNode(returns[1]))
    body = :($rvalues = $(f)($(fargs...)))
  elseif nr > 1
    rvalues = Expr(:tuple, [Expr(:., :_state, QuoteNode(returns[i])) for i in 1:nr]...)
    body = :($rvalues = $(f)($(fargs...)))
  else
    error("Vector of return symbols must have one or more elements")
  end

  args = [statetype == nothing ? :_state : :(_state::$statetype)]
  if states
    push!(args, :(_states::VariableStateVector))
  end

  @gensym lowlevel_variable_method

  quote
    function $lowlevel_variable_method($(args...))
      $body
    end
  end
end
