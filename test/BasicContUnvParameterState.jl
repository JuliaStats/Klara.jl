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

println("    Testing BasicContUnvParameterState constructors and methods...")

v = Float64(-9.2)
s = BasicContUnvParameterState(v)

@test eltype(s) == Float64
@test s.value == v
for i in 2:13
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == []
@test s.diagnostickeys == Symbol[]

@test diagnostics(s) == Dict{Symbol, Any}()

v = Float64(1.5)
dv = [true]
s = BasicContUnvParameterState(v, [:accept], nothing, nothing, dv)

@test eltype(s) == Float64
@test s.value == v
for i in 2:13
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == dv
@test s.diagnostickeys == [:accept]

@test diagnostics(s) == Dict(:accept => dv[1])

s = BasicContUnvParameterState()

@test isa(s.value, Float64)
for i in 1:13
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == []
@test s.diagnostickeys == Symbol[]

@test diagnostics(s) == Dict{Symbol, Any}()

dv = [false]
s = BasicContUnvParameterState([:accept], Float16, nothing, nothing, dv)

@test isa(s.value, Float16)
for i in 1:13
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == dv
@test s.diagnostickeys == [:accept]

@test diagnostics(s) == Dict(:accept => dv[1])

z = BasicContUnvParameterState(Float64(-3.1), [:accept], nothing, nothing, [false])
w = deepcopy(z)
@test isequal(z, w)
