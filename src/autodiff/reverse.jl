function codegen_reverse_autodiff_function(f::Expr, init::Tuple, order::Int, allorders::Bool=true)
  @gensym reverse_autodiff_function
  Expr(
    :function,
    Expr(:call, reverse_autodiff_function, Expr(:(::), init[1], :Vector)),
    eval(Expr(
      :call,
      :(ReverseDiffSource.rdiff),
      :($(Expr(:quote, f))),
      Expr(:kw, init...),
      Expr(:kw, :order, order),
      Expr(:kw, :allorders, allorders))
    )
  )
end
