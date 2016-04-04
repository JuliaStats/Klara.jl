function codegen_setfield(parameter::Parameter, field::Symbol, f::Function)
  @gensym setfield
  quote
    function $setfield(_state::$(default_state_type(parameter)), _states::VariableStateVector)
      setfield!($(parameter), $(QuoteNode(field)), $(f)(_state, _states))
    end
  end
end

function codegen_closure(parameter::Parameter, f::Function)
  @gensym method
  quote
    function $method(_state::$(default_state_type(parameter)))
      $(f)(_state, $(parameter).states)
    end
  end
end

function codegen_closure_via_distribution(parameter::Parameter, distribution::Symbol, f::Function, field::Symbol)
  @gensym method_via_distribution
  quote
    function $method_via_distribution(_state::$(default_state_type(parameter)))
      setfield!(_state, $(QuoteNode(field)), $(f)(getfield($(parameter), $(QuoteNode(distribution))), _state.value))
    end
  end
end

function codegen_closure_via_sum(
  parameter::Parameter, plfield::Symbol, ppfield::Symbol, stfield::Symbol, slfield::Symbol, spfield::Symbol
)
  body = []

  push!(body, :(getfield($(parameter), $(QuoteNode(plfield)))(_state)))
  push!(body, :(getfield($(parameter), $(QuoteNode(ppfield)))(_state)))
  push!(body, :(setfield!(
    _state,
    $(QuoteNode(stfield)),
    getfield(_state, $(QuoteNode(slfield)))+getfield(_state, $(QuoteNode(spfield)))))
  )

  @gensym method_via_sum

  quote
    function $method_via_sum(_state::$(default_state_type(parameter)))
      $(body...)
    end
  end
end

function codegen_uptomethods(parameter::Parameter, fields::Vector{Symbol})
  body = []
  local f::Symbol

  for i in 1:length(fields)
    f = fields[i]
    push!(body, :(getfield($(parameter), $(QuoteNode(f)))(_state)))
  end

  @gensym uptomethods

  quote
    function $uptomethods(_state::$(default_state_type(parameter)))
      $(body...)
    end
  end
end
