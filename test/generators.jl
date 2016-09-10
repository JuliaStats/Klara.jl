using Klara

println("    Testing likelihood_model generator...")

θ = BasicContUnvParameter(:θ, 1, signature=:low)
x = Data(:x, 2)
λ = Hyperparameter(:λ, 3)

likelihood_model([θ, x, λ])
