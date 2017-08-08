mutable struct GenericModel
  is_directed::Bool
  vertices::Vector{Variable}           # An indexable container of vertices (variables)
  edges::Vector{Dependence}            # An indexable container of edges (dependencies)
  finclist::Vector{Vector{Dependence}} # Forward incidence list
  binclist::Vector{Vector{Dependence}} # Backward incidence list
  ofkey::Dict{Symbol, Integer}         # Dictionary storing index of vertex (variable) of corresponding key
end

getindex(m::GenericModel, k::Symbol) = m.vertices[m.ofkey[k]]

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

keys(m::GenericModel) = keys(m.vertices)
indexes(m::GenericModel) = indexes(m.vertices)

out_edges(v::Variable, m::GenericModel) = m.finclist[vertex_index(v, m)]
out_degree(v::Variable, m::GenericModel) = length(out_edges(v, m))

in_edges(v::Variable, m::GenericModel) = m.binclist[vertex_index(v, m)]
in_degree(v::Variable, m::GenericModel) = length(in_edges(v, m))

function add_vertex!(m::GenericModel, v::Variable, n::Integer=num_vertices(m)+1)
    push!(m.vertices, v)

    push!(m.finclist, Integer[])
    push!(m.binclist, Integer[])

    m.ofkey[v.key] = n

    v
end

function add_vertex!(m::GenericModel, vs::VariableVector, n::Integer=num_vertices(m)+1)
  nvertices = n
  for v in vs
    add_vertex!(m, v, nvertices)
    nvertices += 1
  end
end

function set_vertex!(m::GenericModel, v::Variable)
  m.vertices[v.index] = v

  push!(m.finclist, Integer[])
  push!(m.binclist, Integer[])

  m.ofkey[v.key] = v.index

  v
end

function set_vertex!(m::GenericModel, vs::VariableVector)
  for v in vs
    set_vertex!(m, v)
  end
end

make_edge(m::GenericModel, s::Variable, t::Variable) = Dependence(num_edges(m)+1, s, t)

function add_edge!(m::GenericModel, u::Variable, v::Variable, d::Dependence)
    ui = vertex_index(u, m)
    vi = vertex_index(v, m)

    push!(m.edges, d)
    push!(m.finclist[ui], d)
    push!(m.binclist[vi], d)

    if !is_directed(m)
        rev_d = revedge(d)
        push!(m.finclist[vi], rev_d)
        push!(m.binclist[ui], rev_d)
    end

    d
end

add_edge!(m::GenericModel, d::Dependence) = add_edge!(m, source(d, m), target(d, m), d)
add_edge!(m::GenericModel, u::Variable, v::Variable) = add_edge!(m, u, v, make_edge(m, u, v))

function GenericModel(vs::VariableVector, ds::DependenceVector=Dependence[]; isdirected::Bool=true, isindexed::Bool=true)
  n = length(vs)

  m = GenericModel(
    isdirected,
    Variable[],
    Dependence[],
    multivecs(Dependence, n),
    multivecs(Dependence, n),
    Dict{Symbol, Integer}()
  )
  if isindexed
    add_vertex!(m, sort_by_index(vs), 1)
  else
    add_vertex!(m, vs, 1)
    for i in 1:n
      m.vertices[i].index = i
    end
  end

  for d in ds
    add_edge!(m, d)
  end

  return m
end

function GenericModel(vs::VariableVector, ds::Vector{Dependence}, isdirected::Bool)
  n = length(vs)

  m = GenericModel(
    isdirected,
    Array{typeof(vs)}(n),
    Dependence[],
    multivecs(Dependence, n),
    multivecs(Dependence, n),
    Dict{Symbol, Integer}()
  )

  set_vertex!(m, vs)

  for d in ds
    add_edge!(m, d)
  end

  return m
end

GenericModel(isdirected::Bool=true) = GenericModel(Variable[], Dependence[], isdirected)

GenericModel(vs::Vector{V}, ds::Matrix{V}; isdirected::Bool=true, isindexed::Bool=true) where {V<:Variable} =
  GenericModel(vs, [Dependence(i, ds[i, 1], ds[i, 2]) for i in 1:size(ds, 1)], isdirected=isdirected, isindexed=isindexed)

function GenericModel(vs::Dict{Symbol, DataType}, ds::Dict{Symbol, Symbol}, isdirected::Bool=true)
  i = 0
  m = GenericModel(Variable[vs[k](k, i+=1) for k in keys(vs)], Dependence[], isdirected, true)

  i = 0
  for k in keys(ds)
    add_edge!(m, Dependence(i+=1, m[k], m[ds[k]]))
  end

  return m
end

function show(io::IO, model::GenericModel)
  isdirected = is_directed(model) ? "directed": "undirected"
  print(io, "GenericModel: $(num_vertices(model)) variables, $(num_edges(model)) dependencies ($isdirected graph)")
end

function model2dot(stream::IOStream, model::GenericModel)
  graphkeyword, edgesign = is_directed(model) ? ("digraph", "->") : ("graph", "--")
  dotindentation, dotspacing = "  ", " "

  write(stream, "$graphkeyword GenericModel {\n")

  for v in vertices(model)
    write(stream, string(dotindentation, v.key, dotspacing, "[shape=", dotshape(v), "]\n"))
  end

  for d in edges(model)
    write(stream, string(dotindentation, d.source.key, dotspacing, edgesign, dotspacing, d.target.key, "\n"))
  end

  write(stream, "}\n")
end

function model2dot(filename::AbstractString, model::GenericModel, mode::AbstractString="w")
  stream = open(filename, mode)
  model2dot(stream, model)
  close(stream)
end
