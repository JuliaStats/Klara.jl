using Base.Test
using Lora

println("    Testing logistic function...")

@test logistic(0.7, 3, 4, 2.1, 1.4) == 1.4110527196983078
