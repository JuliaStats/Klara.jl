using Base.Test
using Klara

println("    Testing VanillaMCTuner constructors...")

tuner = VanillaMCTuner()

@test tuner.period == 100
@test tuner.verbose == false

tuner = VanillaMCTuner(period=1000, verbose=true)

@test tuner.period == 1000
@test tuner.verbose == true
