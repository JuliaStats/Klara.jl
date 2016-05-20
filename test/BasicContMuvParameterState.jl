using Base.Test
using Lora

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
  @test isnan(s.(fnames[i]))
end
for i in 5:7
  @test length(s.(fnames[i])) == 0
end
for i in 8:10
  @test size(s.(fnames[i])) == (0, 0)
end
for i in 11:13
  @test size(s.(fnames[i])) == (0, 0, 0)
end
@test s.size == length(v)

v = Float16[0.24, 5.5, -6.3]
ssize = length(v)
s = BasicContMuvParameterState(v, [:gradlogtarget], [:accept], [false])

@test eltype(s) == Float16
@test s.value == v
@test length(s.gradlogtarget) == ssize
for i in 2:4
  @test isnan(s.(fnames[i]))
end
for i in 5:6
  @test length(s.(fnames[i])) == 0
end
for i in 8:10
  @test size(s.(fnames[i])) == (0, 0)
end
for i in 11:13
  @test size(s.(fnames[i])) == (0, 0, 0)
end
@test s.diagnosticvalues == [false]
@test s.size == ssize
@test s.diagnostickeys == [:accept]

ssize = 4
s = BasicContMuvParameterState(ssize, Symbol[], Symbol[], BigFloat)

@test isa(s.value, Vector{BigFloat})
@test length(s.value) == ssize
for i in 2:4
  @test isnan(s.(fnames[i]))
end
for i in 5:7
  @test length(s.(fnames[i])) == 0
end
for i in 8:10
  @test size(s.(fnames[i])) == (0, 0)
end
for i in 11:13
  @test size(s.(fnames[i])) == (0, 0, 0)
end
@test s.diagnosticvalues == []
@test s.diagnostickeys == Symbol[]

z = BasicContMuvParameterState(Float64[-0.12, 12.15])
w = deepcopy(z)
@test isequal(z, w)
