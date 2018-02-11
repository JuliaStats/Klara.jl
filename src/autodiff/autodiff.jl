mutable struct DiffMethods
  closurell::Union{Function, Void}
  closurelp::Union{Function, Void}
  closurelt::Union{Function, Void}
  tapegll::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape, Void}
  tapeglp::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape, Void}
  tapeglt::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape, Void}
  tapetll::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape, Void}
  tapetlp::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape, Void}
  tapetlt::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape, Void}
end

DiffMethods(;
  closurell::Union{Function, Void}=nothing,
  closurelp::Union{Function, Void}=nothing,
  closurelt::Union{Function, Void}=nothing,
  tapegll::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape, Void}=nothing,
  tapeglp::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape, Void}=nothing,
  tapeglt::Union{ReverseDiff.GradientTape, ReverseDiff.CompiledTape, Void}=nothing,
  tapetll::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape, Void}=nothing,
  tapetlp::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape, Void}=nothing,
  tapetlt::Union{ReverseDiff.HessianTape, ReverseDiff.CompiledTape, Void}=nothing
) =
 DiffMethods(closurell, closurelp, closurelt, tapegll, tapeglp, tapeglt, tapetll, tapetlp, tapetlt)

mutable struct DiffState
  resultll::Union{DiffResults.DiffResult, Void}
  resultlp::Union{DiffResults.DiffResult, Void}
  resultlt::Union{DiffResults.DiffResult, Void}
  cfggll::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}
  cfgglp::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}
  cfgglt::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}
  cfgtll::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}
  cfgtlp::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}
  cfgtlt::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}
end

DiffState(;
  resultll::Union{DiffResults.DiffResult, Void}=nothing,
  resultlp::Union{DiffResults.DiffResult, Void}=nothing,
  resultlt::Union{DiffResults.DiffResult, Void}=nothing,
  cfggll::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}=nothing,
  cfgglp::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}=nothing,
  cfgglt::Union{ReverseDiff.GradientConfig, ForwardDiff.GradientConfig, Void}=nothing,
  cfgtll::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}=nothing,
  cfgtlp::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}=nothing,
  cfgtlt::Union{ReverseDiff.HessianConfig, ForwardDiff.HessianConfig, Void}=nothing
) =
  DiffState(resultll, resultlp, resultlt, cfggll, cfgglp, cfgglt, cfgtll, cfgtlp, cfgtlt)

==(z::DiffState, w::DiffState) = reduce(&, [getfield(z, n) == getfield(w, n) for n in fieldnames(DiffState)])

isequal(z::DiffState, w::DiffState) = reduce(&, [isequal(getfield(z, n), getfield(w, n)) for n in fieldnames(DiffState)])

mutable struct DiffOptions
  mode::Symbol
  order::Integer
  targets::Vector{Bool}
  chunksize::Integer
  compiled::Bool

  function DiffOptions(mode::Symbol, order::Integer, targets::Vector{Bool}, chunksize::Integer, compiled::Bool)
    @assert (mode == :reverse || mode == :forward) "Mode of automatic differentation must be :reverse or :forward, got $mode"
    @assert (order == 1 || order == 2) "Order of differentiation must be 1 or 2, got order=$order"
    @assert (length(targets) == 3) "Length of targets must be 3, got $length(targets)-length vector"
    @assert chunksize >= 0 "chunksize can not be negative, got chunksize=$chunksize"
    new(mode, order, targets, chunksize, compiled)
  end
end

DiffOptions(mode::Symbol, order::Integer) = DiffOptions(mode, order, fill(false, 3), 0, mode == :reverse ? true : false)

DiffOptions(;
  mode::Symbol=:reverse, order::Integer=1, targets::Vector{Bool}=fill(false, 3), chunksize::Integer=0, compiled::Bool=true
) =
  DiffOptions(mode, order, targets, chunksize, compiled)
