type GenericModel <: AbstractGraph{Variable, Dependence}
  is_directed::Bool
  vertices::Vector{Variable}           # An indexable container of vertices (variables)
  edges::Vector{Dependence}            # An indexable container of edges (dependencies)
  finclist::Vector{Vector{Dependence}} # Forward incidence list
  binclist::Vector{Vector{Dependence}} # Backward incidence list
  indexof::Dict{Variable, Int}         # Dictionary storing index of vertex (variable)
  ofkey::Dict{Symbol, Int}             # Dictionary storing vertex (variable) index of corresponding key
end

@graph_implements GenericModel vertex_list edge_list

Base.getindex(m::GenericModel, k::Symbol) = m.vertices[m.ofkey[k]]

is_directed(m::GenericModel) = m.is_directed

num_vertices(m::GenericModel) = length(m.vertices)
vertices(m::GenericModel) = m.vertices

num_edges(m::GenericModel) = length(m.edges)
edges(m::GenericModel) = m.edges

vertex_index(v::Integer, m::GenericModel) = (v <= m.vertices[end] ? v : 0)
vertex_index(v::Variable, m::GenericModel) = vertex_index(v)

edge_index(d::Dependence, m::GenericModel) = edge_index(d)
source(d::Dependence, m::GenericModel) = d.source
target(d::Dependence, m::GenericModel) = d.target

Base.keys(m::GenericModel) = keys(m.vertices)

out_edges(v::Variable, m::GenericModel) = m.finclist[vertex_index(v, m)]
out_degree(v::Variable, m::GenericModel) = length(out_edges(v, m))
out_neighbors(v::Variable, m::GenericModel) = Graphs.TargetIterator(m, out_edges(v, m))

in_edges(v::Variable, m::GenericModel) = m.binclist[vertex_index(v, m)]
in_degree(v::Variable, m::GenericModel) = length(in_edges(v, m))
in_neighbors(v::Variable, m::GenericModel) = Graphs.SourceIterator(m, in_edges(v, m))

make_edge(m::GenericModel, s::Variable, t::Variable) = Dependence(num_edges(m)+1, s, t)

function add_vertex!(m::GenericModel, v::Variable)
    push!(m.vertices, v)
    push!(m.finclist, Int[])
    push!(m.binclist, Int[])
    m.indexof[v] = length(m.vertices)
    m.ofkey[v.key] = m.indexof[v]
    v
end

function add_edge!(m::GenericModel, u::Variable, v::Variable, d::Dependence)
    ui = vertex_index(u, m)::Int
    vi = vertex_index(v, m)::Int

    push!(m.edges, d)
    push!(m.finclist[ui], d)
    push!(m.binclist[vi], d)

    if !m.is_directed
        rev_d = revedge(d)
        push!(m.finclist[vi], rev_d)
        push!(m.binclist[ui], rev_d)
    end

    d
end

add_edge!(m::GenericModel, d::Dependence) = add_edge!(m, source(d, m), target(d, m), d)
add_edge!(m::GenericModel, u::Variable, v::Variable) = add_edge!(m, u, v, make_edge(m, u, v))

function GenericModel{V<:Variable}(vs::Vector{V}, ds::Vector{Dependence}, is_directed::Bool=true)
  n = length(vs)
  m = GenericModel(
    is_directed,
    Variable[],
    Dependence[],
    Graphs.multivecs(Dependence, n),
    Graphs.multivecs(Dependence, n),
    Dict{Variable, Int}(),
    Dict{Symbol, Variable}()
  )

  for v in vs
    add_vertex!(m, v)
    m.indexof[v] = v.index
    m.ofkey[v.key] = m.indexof[v]
  end

  for d in ds
    add_edge!(m, d)
  end

  return m
end

GenericModel(is_directed::Bool=true) = GenericModel(Variable[], Dependence[], is_directed)

GenericModel{V<:Variable}(vs::Vector{V}, ds::Matrix{V}, is_directed::Bool=true) =
  GenericModel(vs, [Dependence(i, ds[i, 1], ds[i, 2]) for i in 1:size(ds, 1)], is_directed)

function GenericModel(vs::Dict{Symbol, DataType}, ds::Dict{Symbol, Symbol}, is_directed::Bool=true)
  i = 0
  m = GenericModel(Variable[vs[k](i+=1, k) for k in keys(vs)], Dependence[], is_directed)

  i = 0
  for k in keys(ds)
    add_edge!(m, Dependence(i+=1, m[k], m[ds[k]]))
  end

  return m
end

function Base.convert(::Type{GenericGraph}, m::GenericModel)
  dict = Dict{KeyVertex{Symbol}, Int}()
  for (k, v) in m.indexof
    dict[convert(KeyVertex, k)] = v
  end

  Graph{KeyVertex{Symbol}, Edge{KeyVertex{Symbol}}}(
    m.is_directed,
    convert(Vector{KeyVertex}, m.vertices),
    convert(Vector{Edge}, m.edges),
    Vector{Edge{KeyVertex{Symbol}}}[convert(Vector{Edge}, i) for i in m.finclist],
    Vector{Edge{KeyVertex{Symbol}}}[convert(Vector{Edge}, i) for i in m.binclist],
    dict
  )
end

function topological_sort_by_dfs(m::GenericModel)
  g = convert(GenericGraph, m)
  ngvs = num_vertices(g)
  mvs = Array(Variable, ngvs)

  gvs = topological_sort_by_dfs(g)

  for i in 1:ngvs
    mvs[i] = m.vertices[gvs[i].index]
  end

  mvs
end
