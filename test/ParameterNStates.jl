using Base.Test
using Klara

fnames = (
  :value,
  :loglikelihood,
  :logprior,
  :logtarget,
  :gradloglikelihood,
  :gradlogprior,
  :gradlogtarget,
  :tensorloglikelihood,
  :tensorlogprior,
  :tensorlogtarget,
  :dtensorloglikelihood,
  :dtensorlogprior,
  :dtensorlogtarget
)

println("    Testing ContUnvMarkovChain constructors and methods...")

nstaten = 4
nstate = ContUnvMarkovChain(nstaten)

@test eltype(nstate) == Float64
@test length(nstate.value) == nstaten
for i in 2:13
  @test length(getfield(nstate, fnames[i])) == 0
end
@test size(nstate.diagnosticvalues) == (0, 0)
@test nstate.n == nstaten
@test length(nstate.diagnostickeys) == 0

nstaten = 5
nstate = ContUnvMarkovChain(nstaten, [true; fill(false, 5); true; fill(false, 6)], [:accept], Float32)

@test eltype(nstate) == Float32
@test length(nstate.value) == nstaten
@test length(nstate.gradlogtarget) == nstaten
for i in [2:6; 8:13]
  @test length(getfield(nstate, fnames[i])) == 0
end
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.n == nstaten
@test length(nstate.diagnostickeys) == 1

statev = Float32(3.)
stateglt = Float32(4.21)
state = BasicContUnvParameterState(statev, [:accept], nothing, nothing, [true])
state.gradlogtarget = stateglt
savei = 2

copy!(nstate, state, savei)
@test nstate.value[savei] == statev
@test nstate.gradlogtarget[savei] == stateglt
nstate.diagnosticvalues[savei] == true

nstaten = 10
nstate = ContUnvMarkovChain(nstaten, [:value, :logtarget], [:accept])

@test eltype(nstate) == Float64
@test length(nstate.value) == nstaten
@test length(nstate.logtarget) == nstaten
for i in [2:3; 5:13]
  @test length(getfield(nstate, fnames[i])) == 0
end
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.n == nstaten
@test length(nstate.diagnostickeys) == 1

statev = Float64(1.25)
statelt = Float64(-1.12)
state = BasicContUnvParameterState(statev, [:accept], nothing, nothing, [false])
state.logtarget = statelt
savei = 7

copy!(nstate, state, savei)
@test nstate.value[savei] == statev
@test nstate.logtarget[savei] == statelt
nstate.diagnosticvalues[savei] == false

nstatev = Float64[4.39, -9,47]
z = ContUnvMarkovChain(length(nstatev))
z.value = copy(nstatev)
w = deepcopy(z)
@test isequal(z, w)

println("    Testing ContMuvMarkovChain constructors and methods...")

nstatesize = 2
nstaten = 4
nstate = ContMuvMarkovChain(nstatesize, nstaten)

@test eltype(nstate) == Float64
@test size(nstate.value) == (nstatesize, nstaten)
for i in (2, 3, 4)
  @test length(getfield(nstate, fnames[i])) == 0
end
for i in 5:7
  @test size(getfield(nstate, fnames[i])) == (0, 0)
end
for i in 8:10
  @test size(getfield(nstate, fnames[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(getfield(nstate, fnames[i])) == (0, 0, 0, 0)
end
@test size(nstate.diagnosticvalues) == (0, 0)
@test nstate.size == nstatesize
@test nstate.n == nstaten
@test length(nstate.diagnostickeys) == 0

nstatesize = 2
nstaten = 5
nstate = ContMuvMarkovChain(nstatesize, nstaten, [true; fill(false, 3); true; fill(false, 8)], [:accept], Float32)

@test eltype(nstate) == Float32
@test size(nstate.value) == (nstatesize, nstaten)
@test size(nstate.gradloglikelihood) == (nstatesize, nstaten)
for i in (2, 3, 4)
  @test length(getfield(nstate, fnames[i])) == 0
end
for i in (6, 7)
  @test size(getfield(nstate, fnames[i])) == (0, 0)
end
for i in 8:10
  @test size(getfield(nstate, fnames[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(getfield(nstate, fnames[i])) == (0, 0, 0, 0)
end
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.size == nstatesize
@test nstate.n == nstaten
@test length(nstate.diagnostickeys) == 1

statev = Float32[0.17, 9.44]
stategll = Float32[-0.01, 4.7]
state = BasicContMuvParameterState(statev, [:gradloglikelihood], [:accept], nothing, nothing, [false])
state.gradloglikelihood = stategll
savei = 3

copy!(nstate, state, savei)
@test nstate.value[:, savei] == statev
@test nstate.gradloglikelihood[:, savei] == stategll
nstate.diagnosticvalues[savei] == false

nstatesize = 2
nstaten = 10
nstate = ContMuvMarkovChain(nstatesize, nstaten, [:value, :logtarget, :gradlogtarget], [:accept], Float16)

@test eltype(nstate) == Float16
@test size(nstate.value) == (nstatesize, nstaten)
@test length(nstate.logtarget) == nstaten
@test size(nstate.gradlogtarget) == (nstatesize, nstaten)
for i in (2, 3)
  @test length(getfield(nstate, fnames[i])) == 0
end
for i in (5, 6)
  @test size(getfield(nstate, fnames[i])) == (0, 0)
end
for i in 8:10
  @test size(getfield(nstate, fnames[i])) == (0, 0, 0)
end
for i in 11:13
  @test size(getfield(nstate, fnames[i])) == (0, 0, 0, 0)
end
@test size(nstate.diagnosticvalues) == (1, nstaten)
@test nstate.size == nstatesize
@test nstate.n == nstaten
@test length(nstate.diagnostickeys) == 1

statev = Float16[6.91, 0.42]
statelt = Float16(4.67)
stateglt = Float16[-0.01, 3.2]
state = BasicContMuvParameterState(statev, [:gradlogtarget], [:accept], nothing, nothing, [true])
state.logtarget = statelt
state.gradlogtarget = stateglt
savei = 7

copy!(nstate, state, savei)
@test nstate.value[:, savei] == statev
@test nstate.logtarget[savei] == statelt
@test nstate.gradlogtarget[:, savei] == stateglt
nstate.diagnosticvalues[savei] == true

nstatev = Float64[4.3 9.2 -7.44; -0.2 8.1 4.43]
z = ContMuvMarkovChain(size(nstatev)...)
z.value = copy(nstatev)
w = deepcopy(z)
@test isequal(z, w)
