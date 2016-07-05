codegen_reverse_autodiff_target(method::Symbol, f::Function, fargtype::Symbol, init) =
  codegen_reverse_autodiff_target(Val{method}, f, fargtype, init)

function codegen_reverse_autodiff_target(::Type{Val{:hessian}}, f::Function, fargtype::Symbol, init::Tuple)
  adfunction = ReverseDiffSource.rdiff(f, (init[2],), order=2, allorders=false)

  @gensym reverse_autodiff_target
  quote
    function $reverse_autodiff_target($(init[1])::$fargtype)
      -$(adfunction)($(init[1]))
    end
  end
end

function codegen_reverse_autodiff_target(::Type{Val{:hessian}}, f::Function, fargtype::Symbol, init::Vector)
  adfunction = ReverseDiffSource.rdiff(f, (init[1][2], init[2][2]), ignore=[init[2][1]], order=2, allorders=false)

  @gensym reverse_autodiff_target
  quote
    function $reverse_autodiff_target($(init[1][1])::$fargtype, $(init[2][1])::Vector)
      -$(adfunction)($(init[1][1]), $(init[2][1]))
    end
  end
end

codegen_reverse_autodiff_uptotarget(method::Symbol, f::Function, fargtype::Symbol, init) =
  codegen_reverse_autodiff_uptotarget(Val{method}, f, fargtype, init)

function codegen_reverse_autodiff_uptotarget(::Type{Val{:hessian}}, f::Function, fargtype::Symbol, init::Tuple)
  adfunction = ReverseDiffSource.rdiff(f, (init[2],), order=2, allorders=true)

  @gensym reverse_autodiff_uptotarget
  quote
    function $reverse_autodiff_uptotarget($(init[1])::$fargtype)
      v, g, h = $(adfunction)($(init[1]))
      return v, g, -h
    end
  end
end

function codegen_reverse_autodiff_uptotarget(::Type{Val{:hessian}}, f::Function, fargtype::Symbol, init::Vector)
  adfunction = ReverseDiffSource.rdiff(f, (init[1][2], init[2][2]), ignore=[init[2][1]], order=2, allorders=true)

  @gensym reverse_autodiff_uptotarget
  quote
    function $reverse_autodiff_uptotarget($(init[1][1])::$fargtype, $(init[2][1])::Vector)
      v, g, h = $(adfunction)($(init[1][1]), $(init[2][1]))
      return v, g, -h
    end
  end
end

function codegen_reverse_autodiff_function(f::Expr, fargtype::Symbol, init::Tuple, order::Integer, allorders::Bool=true)
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

function codegen_reverse_autodiff_function(f::Expr, fargtype::Symbol, init::Vector, order::Integer, allorders::Bool=true)
  @gensym reverse_autodiff_function
  Expr(
    :function,
    Expr(:call, reverse_autodiff_function, Expr(:(::), init[1][1], fargtype), Expr(:(::), init[2][1], :Vector)),
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

codegen_reverse_autodiff_target(method::Symbol, f::Expr, fargtype::Symbol, init) =
  codegen_reverse_autodiff_target(Val{method}, f, fargtype, init)

function codegen_reverse_autodiff_target(::Type{Val{:hessian}}, f::Expr, fargtype::Symbol, init::Tuple)
  @gensym reverse_autodiff_target
  Expr(
    :function,
    Expr(:call, reverse_autodiff_target, Expr(:(::), init[1], fargtype)),
    Expr(
      :call,
      :-,
      eval(Expr(
        :call,
        :(ReverseDiffSource.rdiff),
        :($(Expr(:quote, f))),
        Expr(:kw, init...),
        Expr(:kw, :order, 2),
        Expr(:kw, :allorders, false)
      ))
    )
  )
end

function codegen_reverse_autodiff_target(::Type{Val{:hessian}}, f::Expr, fargtype::Symbol, init::Vector)
  @gensym reverse_autodiff_target
  Expr(
    :function,
    Expr(:call, reverse_autodiff_target, Expr(:(::), init[1][1], fargtype), Expr(:(::), init[2][1], :Vector)),
    Expr(
      :call,
      :-,
      eval(Expr(
        :call,
        :(ReverseDiffSource.rdiff),
        :($(Expr(:quote, f))),
        Expr(:kw, init[1]...),
        Expr(:kw, init[2]...),
        Expr(:kw, :ignore, [init[2][1]]),
        Expr(:kw, :order, 2),
        Expr(:kw, :allorders, false),
      ))
    )
  )
end

codegen_reverse_autodiff_uptotarget(method::Symbol, f::Expr, fargtype::Symbol, init) =
  codegen_reverse_autodiff_uptotarget(Val{method}, f, fargtype, init)

function codegen_reverse_autodiff_uptotarget(::Type{Val{:hessian}}, f::Expr, fargtype::Symbol, init::Tuple)
  adfunction = eval(Expr(
    :call,
    :(ReverseDiffSource.rdiff),
    :($(Expr(:quote, f))),
    Expr(:kw, init...),
    Expr(:kw, :order, 2),
    Expr(:kw, :allorders, true)
  ))

  @gensym reverse_autodiff_uptotarget
  quote
    function $reverse_autodiff_uptotarget($(init[1])::$fargtype)
      v, g, h = $(adfunction)($(init[1]))
      return v, g, -h
    end
  end
end

function codegen_reverse_autodiff_uptotarget(::Type{Val{:hessian}}, f::Expr, fargtype::Symbol, init::Vector)
  adfunction = eval(Expr(
    :call,
    :(ReverseDiffSource.rdiff),
    :($(Expr(:quote, f))),
    Expr(:kw, init[1]...),
    Expr(:kw, init[2]...),
    Expr(:kw, :ignore, [init[2][1]]),
    Expr(:kw, :order, 2),
    Expr(:kw, :allorders, true),
  ))

  @gensym reverse_autodiff_uptotarget
  quote
    function $reverse_autodiff_uptotarget($(init[1][1])::$fargtype, $(init[2][1])::Vector)
      v, g, h = $(adfunction)($(init[1][1]), $(init[2][1]))
      return v, g, -h
    end
  end
end
