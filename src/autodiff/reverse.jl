function codegen_reverse_autodiff_function(f::Expr, fargtype::Symbol, init::Tuple, order::Int, allorders::Bool=true)
  @gensym reverse_autodiff_function
  Expr(
    :function,
    Expr(:call, reverse_autodiff_function, Expr(:(::), init[1], fargtype)),
    eval(Expr(
      :call,
      :(ReverseDiffSource.rdiff),
      :($(Expr(:quote, f))),
      Expr(:kw, init...),
      Expr(:kw, :order, order),
      Expr(:kw, :allorders, allorders)
    ))
  )
end

function codegen_reverse_autodiff_function(f::Expr, fargtype::Symbol, init::Vector, order::Int, allorders::Bool=true)
  @gensym reverse_autodiff_function
  Expr(
    :function,
    Expr(:call, reverse_autodiff_function, Expr(:(::), init[1][1], fargtype), Expr(:(::), init[2][1], fargtype)),
    eval(Expr(
      :call,
      :(ReverseDiffSource.rdiff),
      :($(Expr(:quote, f))),
      Expr(:kw, init[1]...),
      Expr(:kw, init[2]...),
      Expr(:kw, :ignore, [init[2][1]]),
      Expr(:kw, :order, order),
      Expr(:kw, :allorders, allorders),
    ))
  )
end
