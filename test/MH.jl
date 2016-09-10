using Base.Test
using Klara

println("    Testing MHState constructors...")

v = Float32(1.5)
pstate = BasicContUnvParameterState(v, [:accept], [true])
sstate = MHState(deepcopy(pstate))

@test eltype(sstate) == BasicContUnvParameterState{eltype(v)}
@test isequal(sstate.pstate, pstate)
@test sstate.tune.accepted == 0
@test sstate.tune.proposed == 0
@test isnan(sstate.tune.rate)
@test isnan(sstate.ratio)

v = Float64[-6.55, 2.8]
pstate = BasicContMuvParameterState(v)
sstate = MHState(deepcopy(pstate), VanillaMCTune(10, 100, 0.1))

@test eltype(sstate) == BasicContMuvParameterState{eltype(v)}
@test isequal(sstate.pstate, pstate)
@test sstate.tune.accepted == 10
@test sstate.tune.proposed == 100
@test sstate.tune.rate == 0.1
@test isnan(sstate.ratio)

v = Float16[3.16, -2.97, -8.53]
pstate = BasicContMuvParameterState(v)
sstate = MHState(deepcopy(pstate), VanillaMCTune(), 0.27)

@test eltype(sstate) == BasicContMuvParameterState{eltype(v)}
@test isequal(sstate.pstate, pstate)
@test sstate.tune.accepted == 0
@test sstate.tune.proposed == 0
@test isnan(sstate.tune.rate)
@test sstate.ratio == 0.27
