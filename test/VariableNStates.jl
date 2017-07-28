using Base.Test
using Klara

println("    Testing BasicUnvVariableNState constructors and methods...")

nstatev = Float64[1.25, -4.4, 7.5]
nstate = BasicUnvVariableNState(nstatev)

@test eltype(nstate) == Float64
@test nstate.value == nstatev
@test nstate.n == length(nstatev)

statev = Float64(5.2)
savei = 2

copy!(nstate, BasicUnvVariableState(statev), savei)
@test nstate.value == [nstatev[1:savei-1]; statev; nstatev[savei+1:end]]

nstatev = Float32[-2.17, 1.92, -0.15, -0.65]
nstate = BasicUnvVariableNState(nstatev)

@test eltype(nstate) == Float32
@test nstate.value == nstatev
@test nstate.n == length(nstatev)

statev = Float32(-2.12)
savei = 4

copy!(nstate, BasicUnvVariableState(statev), savei)
@test nstate.value == [nstatev[1:savei-1]; statev; nstatev[savei+1:end]]

nstaten = 10
nstate = BasicUnvVariableNState(nstaten, BigFloat)

@test eltype(nstate) == BigFloat
@test nstate.n == nstaten

println("    Testing BasicMuvVariableNState constructors and methods...")

nstatev = Float64[1.35 3.7 4.5; 5.6 8.81 9.2]
nstate = BasicMuvVariableNState(nstatev)

@test eltype(nstate) == Float64
@test nstate.value == nstatev
@test nstate.size == size(nstatev, 1)
@test nstate.n == size(nstatev, 2)

statev = Float64[5.2, 3.31]
savei = 2

copy!(nstate, BasicMuvVariableState(statev), savei)
@test nstate.value == [nstatev[:, 1:savei-1] statev nstatev[:, savei+1:end]]

nstatev= BigFloat[
  0.41257   0.106756   0.817916   0.569789  0.54802;
  0.630804  0.0212354  0.0729593  0.483741  0.596365;
  0.82968   0.4872     0.185226   0.354095  0.944551;
  0.634923  0.448942   0.300905   0.243899  0.126606
]
nstate = BasicMuvVariableNState(nstatev)

@test eltype(nstate) == BigFloat
@test nstate.value == nstatev
@test nstate.size == size(nstatev, 1)
@test nstate.n == size(nstatev, 2)

statev = BigFloat[0.0646775, 0.379354, 0.0101067, 0.821756]
savei = 1

copy!(nstate, BasicMuvVariableState(BigFloat[0.0646775, 0.379354, 0.0101067, 0.821756]), 1)
@test nstate.value == [nstatev[:, 1:savei-1] statev nstatev[:, savei+1:end]]

nstatesize = 3
nstaten = 10
nstate = BasicMuvVariableNState(nstatesize, nstaten, Float16)

@test eltype(nstate) == Float16
@test nstate.size == nstatesize
@test nstate.n == nstaten

println("    Testing BasicMavVariableNState constructors and methods...")

nstatesize = (2, 4)
nstaten = 5
nstatev = Array{Float64}(nstatesize..., nstaten)
nstatev[:, :, 1] = Float64[
   0.680789  -0.194683   1.86498    0.490497;
  -0.730417   0.305873  -0.0434663  0.879241
]
nstatev[:, :, 2] = Float64[
  -1.63194  -0.257043  -0.981173  0.524005;
  -1.61526   0.173245  -0.677052  1.88443
]
nstatev[:, :, 3] = Float64[
  -0.131442  -1.42516   -0.267594  -1.45806;
   0.544324  -0.109355  -0.219908   1.07273
]
nstatev[:, :, 4] = Float64[
  0.134486   2.0883    -0.455804  -2.20976;
  0.468841  -0.552023  -0.837046  -0.183255
]
nstatev[:, :, 5] = Float64[
  -1.3907   -1.18854  0.985564   0.373107;
   1.21794  -1.04891  0.611239  -0.526945
]
nstate = BasicMavVariableNState(nstatev)

@test eltype(nstate) == Float64
@test nstate.value == nstatev
@test nstate.size == nstatesize
@test nstate.n == nstaten

statev = Float64[
   1.36032    0.200834  -1.83856  -1.3039;
  -0.641921  -1.31766    1.35137   0.938878
]
savei = 3

copy!(nstate, BasicMavVariableState(statev), savei)
for i in [1:savei-1; savei; savei+1:nstaten]
  @test nstate.value[:, :, i] == nstatev[:, :, i]
end
@test nstate.value[:, :, savei] == statev

nstatesize = (3, 4)
nstaten = 2
nstatev = Array{Float16}(nstatesize..., nstaten)
nstatev[:, :, 1] = Float16[
  -0.372612   0.00410037  -1.11811   -0.278956;
  -0.744771  -1.63419     -2.12376    1.161;
   1.0475    -1.9372       0.198637  -1.45228
]
nstatev[:, :, 2] = Float16[
  -0.124304  -1.60967   -0.0316585  -1.06222;
  -1.18789    0.849805   0.442735    0.454251;
   0.356338   0.60818    0.0970741   0.433813
]
nstate = BasicMavVariableNState(nstatev)

@test eltype(nstate) == Float16
@test nstate.value == nstatev
@test nstate.size == nstatesize
@test nstate.n == nstaten

statev = Float16[
   0.538734  -2.18586     1.4828     0.0479034;
   0.225186   0.28757    -0.580649   0.535829;
  -1.92379    0.0638309  -0.205978  -0.604045
]
savei = 2

copy!(nstate, BasicMavVariableState(statev), savei)
for i in [1:savei-1; savei; savei+1:nstaten]
  @test nstate.value[:, :, i] == nstatev[:, :, i]
end
@test nstate.value[:, :, savei] == statev

nstatesize = (3, 5)
nstaten = 12
nstate = BasicMavVariableNState(nstatesize, nstaten, BigFloat)

@test eltype(nstate) == BigFloat
@test nstate.size == nstatesize
@test nstate.n == nstaten
