using Lora

println("    Testing likelihood_model generator...")

θ = BasicContUnvParameter(:θ, 1)
x = Data(:x, 2)
λ = Hyperparameter(:λ, 3)

likelihood_model([θ], data=[x], hyperparameters=[λ])

println("    Testing single_parameter_likelihood_model generator...")

single_parameter_likelihood_model(θ, data=[x], hyperparameters=[λ])
