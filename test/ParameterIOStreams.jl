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
  :dtensorlogtarget,
  :diagnosticvalues
)
filepath = dirname(@__FILE__)
filesuffix = "csv"
filenames = Array{AbstractString}(14)
for i in 1:14
  filenames[i] = joinpath(filepath, string(fnames[i])*"."*filesuffix)
end

println("    Testing BasicContParamIOStream constructors and close method...")

iostreamsize = ()
iostreamn = 4
iostream = BasicContParamIOStream(iostreamsize, iostreamn, filepath=filepath)

@test isa(iostream.value, IOStream)
for i in 2:14
  @test getfield(iostream, fnames[i]) == nothing
end
@test iostream.size == iostreamsize
@test iostream.n == iostreamn
@test length(iostream.diagnostickeys) == 0

close(iostream)
rm(filenames[1])

iostreamsize = (2,)
iostreamn = 4
iostream = BasicContParamIOStream(
  iostreamsize,
  iostreamn,
  monitor=[true; fill(false, 5); true; fill(false, 6)],
  filepath=filepath,
  diagnostickeys=[:accept]
)

for i in (1, 7, 14)
  @test isa(getfield(iostream, fnames[i]), IOStream)
end
for i in [2:6; 8:13]
  @test getfield(iostream, fnames[i]) == nothing
end
@test iostream.size == iostreamsize
@test iostream.n == iostreamn
@test length(iostream.diagnostickeys) == 1

close(iostream)
for i in (1, 7, 14); rm(filenames[i]); end

iostreamsize = (3,)
iostreamn = 7
iostream = BasicContParamIOStream(iostreamsize, iostreamn, [:value, :logtarget], filepath=filepath, diagnostickeys=[:accept])

for i in (1, 4, 14)
  @test isa(getfield(iostream, fnames[i]), IOStream)
end
for i in [2:3; 5:13]
  @test getfield(iostream, fnames[i]) == nothing
end
@test iostream.size == iostreamsize
@test iostream.n == iostreamn
@test length(iostream.diagnostickeys) == 1

close(iostream)
for i in [1, 4, 14]; rm(filenames[i]); end

println("    Testing BasicContParamIOStream IO methods...")

println("      Interaction with BasicContUnvParameterState...")

nstatev = Float64[5.70, 1.44, -1.21, 5.67]
iostreamsize = ()
iostreamn = length(nstatev)

iostream = BasicContParamIOStream(iostreamsize, iostreamn, filepath=filepath)
for v in nstatev
  write(iostream, BasicContUnvParameterState(v))
end

close(iostream)

iostream = BasicContParamIOStream(iostreamsize, iostreamn, filepath=filepath, mode="r")
nstate = read(iostream, Float64)

@test isa(nstate, ContUnvMarkovChain{Float64})
@test nstate.value == nstatev
for i in 2:13
  @test length(getfield(nstate, fnames[i])) == 0
end
@test length(nstate.diagnostickeys) == 0
@test size(nstate.diagnosticvalues) == (0, 0)
@test nstate.n == iostream.n

close(iostream)
rm(filenames[1])

println("      Interaction with ContUnvMarkovChain...")

nstatev = Float32[1.93, 98.46, -3.61, -0.99, 74.52, 9.90]
nstated = Any[false, true, true, false, true, false]'
iostreamsize = ()
iostreamn = length(nstatev)

iostream = BasicContParamIOStream(iostreamsize, iostreamn, [:value], filepath=filepath, diagnostickeys=[:accept])
nstatein = ContUnvMarkovChain(iostreamn, [:value], [:accept], Float32)
nstatein.value = nstatev
nstatein.diagnosticvalues = nstated
write(iostream, nstatein)

close(iostream)

iostream = BasicContParamIOStream(iostreamsize, iostreamn, [:value], filepath=filepath, diagnostickeys=[:accept], mode="r")
nstateout = read(iostream, Float32)

@test isa(nstateout, ContUnvMarkovChain{Float32})
@test nstateout.value == nstatein.value
for i in 2:13
  @test length(getfield(nstateout, fnames[i])) == 0
end
@test length(nstateout.diagnostickeys) == 1
@test nstateout.diagnosticvalues == nstatein.diagnosticvalues
@test nstateout.n == nstatein.n

close(iostream)
for i in (1, 14); rm(filenames[i]); end

println("      Interaction with BasicContMuvParameterState...")

nstatev = Float64[1.33 2.44 3.14 -0.82; 7.21 -9.75 -5.26 -0.63]
nstategll = Float64[3.13 -12.10 13.11 -0.99; 9.91 -5.25 -8.15 -9.69]
nstated = Any[false, true, true, false]'
iostreamsize = (size(nstatev, 1),)
iostreamn = size(nstatev, 2)

iostream = BasicContParamIOStream(
  iostreamsize, iostreamn, [:value, :gradloglikelihood], filepath=filepath, diagnostickeys=[:accept]
)
for i in 1:iostreamn
  state = BasicContMuvParameterState(nstatev[:, i], Symbol[], [:accept], nothing, nothing, [nstated[i]])
  state.gradloglikelihood = nstategll[:, i]
  write(iostream, state)
end

close(iostream)

iostream = BasicContParamIOStream(
  iostreamsize, iostreamn, [:value, :gradloglikelihood], filepath=filepath, diagnostickeys=[:accept], mode="r"
)
nstate = read(iostream, Float64)

@test isa(nstate, ContMuvMarkovChain{Float64})
@test nstate.value == nstatev
@test nstate.gradloglikelihood == nstategll
for i in [2:4; 6:13]
  @test length(getfield(nstate, fnames[i])) == 0
end
@test length(nstate.diagnostickeys) == 1
@test nstate.diagnosticvalues == nstated
@test nstate.size == iostream.size[1]
@test nstate.n == iostream.n

close(iostream)
for i in (1, 5, 14); rm(filenames[i]); end

println("      Interaction with ContMuvMarkovChain...")

nstatev = Float32[-1.85 -0.09 0.36; -0.45 -0.85 1.91]
nstatell = Float32[-1.30, -1.65, -0.18]
nstatelt = Float32[-0.44, 0.72, -0.21]
nstated = Any[false true; true false; true true]'
iostreamsize = (size(nstatev, 1),)
iostreamn = size(nstatev, 2)

iostream = BasicContParamIOStream(
  iostreamsize, iostreamn, [:value, :loglikelihood, :logtarget], filepath=filepath, diagnostickeys=[:accept]
)
nstatein = ContMuvMarkovChain(iostreamsize[1], iostreamn, [:value, :loglikelihood, :logtarget], [:accept], Float32)
nstatein.value = nstatev
nstatein.loglikelihood = nstatell
nstatein.logtarget = nstatelt
nstatein.diagnosticvalues = nstated
write(iostream, nstatein)

close(iostream)

iostream = BasicContParamIOStream(
  iostreamsize, iostreamn, [:value, :loglikelihood, :logtarget], filepath=filepath, diagnostickeys=[:accept], mode="r"
)
nstateout = read(iostream, Float32)

@test isa(nstateout, ContMuvMarkovChain{Float32})
@test nstateout.value == nstatein.value
@test nstateout.loglikelihood == nstatein.loglikelihood
@test nstateout.logtarget == nstatein.logtarget
for i in [3; 5:13]
  @test length(getfield(nstateout, fnames[i])) == 0
end
@test length(nstateout.diagnostickeys) == 1
@test nstateout.diagnosticvalues == nstatein.diagnosticvalues
@test nstateout.n == nstatein.n

close(iostream)
for i in (1, 2, 4, 14); rm(filenames[i]); end
