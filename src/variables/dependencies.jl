### Dependence

immutable Dependence{S<:Variable, T<:Variable}
  index::Integer
  source::S
  target::T
end

Dependence{S<:Variable, T<:Variable}(source::S, target::T) = Dependence(0, source, target)

typealias DependenceVector{D<:Dependence} Vector{D}

edge_index(d::Dependence) = d.index
source(d::Dependence) = e.source
target(d::Dependence) = e.target

is_indexed(d::Dependence) = d.index > 0 ? true : false

revedge{S<:Variable, T<:Variable}(d::Dependence{S, T}) = Dependence(d.index, d.target, d.source)

Base.show(io::IO, d::Dependence) = print(io, "Dependence [$(d.index)]: $(d.target.key) | $(d.source.key)")
