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

println("    Testing BasicContMuvParameterState constructors and methods...")

v = Float64[1., 1.5]
s = BasicContMuvParameterState(v)

@test eltype(s) == Float64
@test s.value == v
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
for i in 5:7
  @test length(getfield(s, fnames[i])) == 0
end
for i in 8:10
  @test size(getfield(s, fnames[i])) == (0, 0)
end
for i in 11:13
  @test size(getfield(s, fnames[i])) == (0, 0, 0)
end
@test s.diagnosticvalues == []
@test s.size == length(v)
@test s.diagnostickeys == Symbol[]

@test diagnostics(s) == Dict{Symbol, Any}()

v = Float16[0.24, 5.5, -6.3]
dv = [false]
ssize = length(v)
s = BasicContMuvParameterState(v, [:gradlogtarget], [:accept], nothing, nothing, dv)

@test eltype(s) == Float16
@test s.value == v
@test length(s.gradlogtarget) == ssize
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
for i in 5:6
  @test length(getfield(s, fnames[i])) == 0
end
for i in 8:10
  @test size(getfield(s, fnames[i])) == (0, 0)
end
for i in 11:13
  @test size(getfield(s, fnames[i])) == (0, 0, 0)
end
@test s.diagnosticvalues == dv
@test s.size == ssize
@test s.diagnostickeys == [:accept]

@test diagnostics(s) == Dict(:accept => dv[1])

ssize = 3
s = BasicContMuvParameterState(ssize)

@test eltype(s) == Float64
@test isa(s.value, Vector{Float64})
@test length(s.value) == ssize
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
for i in 5:7
  @test length(getfield(s, fnames[i])) == 0
end
for i in 8:10
  @test size(getfield(s, fnames[i])) == (0, 0)
end
for i in 11:13
  @test size(getfield(s, fnames[i])) == (0, 0, 0)
end
@test s.diagnosticvalues == []
@test s.size == ssize
@test s.diagnostickeys == Symbol[]

@test diagnostics(s) == Dict{Symbol, Any}()

dv = [true]
ssize = 5
s = BasicContMuvParameterState(ssize, [:tensorlogtarget], [:accept], BigFloat, nothing, nothing, dv)

@test eltype(s) == BigFloat
@test isa(s.value, Vector{BigFloat})
@test length(s.value) == ssize
@test size(s.tensorlogtarget) == (ssize, ssize)
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
for i in 5:7
  @test length(getfield(s, fnames[i])) == 0
end
for i in 8:9
  @test size(getfield(s, fnames[i])) == (0, 0)
end
for i in 11:13
  @test size(getfield(s, fnames[i])) == (0, 0, 0)
end
@test s.diagnosticvalues == dv
@test s.size == ssize
@test s.diagnostickeys == [:accept]

@test diagnostics(s) == Dict(:accept => dv[1])

z = BasicContMuvParameterState(Float64[-0.12, 12.15], [:gradloglikelihood], [:accept], nothing, nothing, [false])
w = deepcopy(z)
@test isequal(z, w)
