type DiffMethods
  closurell::Union{Function, Void}
  closurelp::Union{Function, Void}
  closurelt::Union{Function, Void}
  tapell::Union{ReverseDiff.AbstractTape, Void}
  tapelp::Union{ReverseDiff.AbstractTape, Void}
  tapelt::Union{ReverseDiff.AbstractTape, Void}
end

DiffMethods(;
  closurell::Union{Function, Void}=nothing,
  closurelp::Union{Function, Void}=nothing,
  closurelt::Union{Function, Void}=nothing,
  tapell::Union{ReverseDiff.AbstractTape, Void}=nothing,
  tapelp::Union{ReverseDiff.AbstractTape, Void}=nothing,
  tapelt::Union{ReverseDiff.AbstractTape, Void}=nothing
) =
 DiffMethods(closurell, closurelp, closurelt, tapell, tapelp, tapelt)

type DiffState
  resultll::Union{DiffBase.DiffResult, Void}
  resultlp::Union{DiffBase.DiffResult, Void}
  resultlt::Union{DiffBase.DiffResult, Void}
  cfggll::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}
  cfgglp::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}
  cfgglt::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}
  cfgtll::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}
  cfgtlp::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}
  cfgtlt::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}
end

DiffState(;
  resultll::Union{DiffBase.DiffResult, Void}=nothing,
  resultlp::Union{DiffBase.DiffResult, Void}=nothing,
  resultlt::Union{DiffBase.DiffResult, Void}=nothing,
  cfggll::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}=nothing,
  cfgglp::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}=nothing,
  cfgglt::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}=nothing,
  cfgtll::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}=nothing,
  cfgtlp::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}=nothing,
  cfgtlt::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}=nothing
) =
  DiffState(resultll, resultlp, resultlt, cfggll, cfgglp, cfgglt, cfgtll, cfgtlp, cfgtlt)

type DiffOptions
  mode::Symbol
  order::Integer
  targets::Vector{Bool}
  chunksize::Integer
end

DiffOptions(mode::Symbol, order::Integer) = DiffOptions(mode, order, fill(false, 3), 0)

DiffOptions(; mode::Symbol=:reverse, order::Integer=1, targets::Vector{Bool}=fill(false, 3), chunksize::Integer=0) =
  DiffOptions(mode, order, targets, chunksize)

codegen_autodiff_function(mode::Symbol, method::Symbol, f::Function) = codegen_autodiff_function(Val{mode}, Val{method}, f)

codegen_autodiff_target(mode::Symbol, method::Symbol, f::Function) = codegen_autodiff_target(Val{mode}, Val{method}, f)

codegen_autodiff_uptofunction(mode::Symbol, method::Symbol, f::Function) =
  codegen_autodiff_uptofunction(Val{mode}, Val{method}, f)

codegen_autodiff_uptotarget(mode::Symbol, method::Symbol, f::Function) =
  codegen_autodiff_uptotarget(Val{mode}, Val{method}, f)
