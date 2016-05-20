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

println("    Testing BasicContUnvParameterState constructors and methods...")

v = Float64(1.5)
s = BasicContUnvParameterState(v, [:accept], [true])

@test eltype(s) == Float64
@test s.value == v
for i in 2:13
  @test isnan(s.(fnames[i]))
end
@test s.diagnosticvalues == [true]
@test s.diagnostickeys == [:accept]

s = BasicContUnvParameterState(Symbol[], Float16)

@test isa(s.value, Float16)
for i in 1:13
  @test isnan(s.(fnames[i]))
end
@test s.diagnosticvalues == []
@test s.diagnostickeys == Symbol[]

z = BasicContUnvParameterState(Float64(-3.1), [:accept], [false])
w = deepcopy(z)
@test isequal(z, w)
