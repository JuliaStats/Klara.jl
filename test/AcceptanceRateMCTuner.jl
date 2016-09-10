using Base.Test
using Klara

println("    Testing score functions for acceptance rate:")

println("      Testing logistic_rate_score...")

@test logistic_rate_score(0.25) == 1.7039056039366212
@test logistic_rate_score(0.5, 11) == 1.991859724568208

println("      Testing erf_rate_score...")

@test erf_rate_score(-0.1) == 0.6713732405408726
@test erf_rate_score(0.93, 2) == 1.9914724883356396

println("    Testing AcceptanceRateMCTuner constructors and methods...")

trate = 0.23
tuner = AcceptanceRateMCTuner(trate)

@test tuner.targetrate == trate
@test tuner.period == 100
@test tuner.verbose == false

tune = AcceptanceRateMCTune(1., 30, 100)
rate!(tune)
orate = tune.rate
tune!(tune, tuner)

@test tune.rate == logistic_rate_score(orate-tuner.targetrate)*orate

trate = 0.45
tuner = AcceptanceRateMCTuner(trate, score=erf_rate_score, period=1000, verbose=true)

@test tuner.targetrate == trate
@test tuner.period == 1000
@test tuner.verbose == true

tune = AcceptanceRateMCTune(1., 621, 1000)
rate!(tune)
orate = tune.rate
tune!(tune, tuner)

@test tune.rate == erf_rate_score(orate-tuner.targetrate)*orate
