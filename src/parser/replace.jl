function replace!(exprn::Expr, operand::Symbol, replacement::Union{Symbol, Expr})
  queue = Array(Union{Number, Symbol, Expr}, 0)
  unshift!(queue, exprn)

  while length(queue) != 0
    node = shift!(queue)

    if isa(node, LineNumberNode)
      continue
    else
      for i in 1:length(node.args)
        if isa(node.args[i], Expr)
          unshift!(queue, node.args[i])
        elseif isa(node.args[i], Symbol) && node.args[i] == operand
          node.args[i] = replacement
        end
      end
    end
  end
end

function replace!(exprn::Expr, operand::Expr, replacement::Union{Symbol, Expr})
  queue = Array(Union{Number, Symbol, Expr}, 0)
  unshift!(queue, exprn)

  while length(queue) != 0
    node = shift!(queue)

    if isa(node, LineNumberNode)
      continue
    else
      for i in 1:length(node.args)
        if isa(node.args[i], Expr)
          if node.args[i] == operand
            node.args[i] = replacement
          else
            unshift!(queue, node.args[i])
          end
        end
      end
    end
  end
end

function replace!(exprn::Expr, dict::Dict{Union{Symbol, Expr}, Union{Symbol, Expr}})
  queue = Array(Union{Number, Symbol, Expr}, 0)
  unshift!(queue, exprn)

  while length(queue) != 0
    node = shift!(queue)

    if isa(node, LineNumberNode)
      continue
    else
      for i in 1:length(node.args)
        if isa(node.args[i], Expr)
          if haskey(dict, node.args[i]) && get(dict, node.args[i], :()) != :()
            node.args[i] = dict[node.args[i]]
          else
            unshift!(queue, node.args[i])
          end
        elseif isa(node.args[i], Symbol) && haskey(dict, node.args[i]) && get(dict, node.args[i], :()) != :()
          node.args[i] = dict[node.args[i]]
        end
      end
    end
  end
end
