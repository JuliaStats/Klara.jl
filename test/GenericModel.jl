using Klara

println("    Testing GenericModel constructors...")

println("      Initialization without input arguments...")

m = GenericModel()

θ = BasicContUnvParameter(:θ, 1, signature=:low)
x = Data(:x, 2)
λ = Hyperparameter(:λ, 3)

add_vertex!(m, θ)
add_vertex!(m, x)
add_vertex!(m, λ)

add_edge!(m, x, θ)
add_edge!(m, λ, θ)

println("      Initialization via matrix-based specification of dependencies...")

m = GenericModel([θ, x, λ], [x θ; λ θ])

# println("    Testing plotting of GenericModel...")

# g = convert(GenericGraph, m)
# plot(g)
