using Base.Test
using Lora

fnames = (
  :value,
  :loglikelihood,
  :logprior,
  :logtarget
)

println("    Testing BasicDiscUnvParameterState constructors and methods...")

v = Int64(-2)
s = BasicDiscUnvParameterState(v)

@test eltype(s) == (Int64, Float64)
@test s.value == v
for i in 2:4
  @test isnan(s.(fnames[i]))
end
@test s.diagnosticvalues == []
@test s.diagnostickeys == Symbol[]

@test diagnostics(s) == Dict{Symbol, Any}()

v = Int32(4)
s = BasicDiscUnvParameterState(v, [:accept], Float32, [true])

@test eltype(s) == (Int32, Float32)
@test s.value == v
for i in 2:4
  @test isnan(s.(fnames[i]))
end
@test s.diagnosticvalues == [true]
@test s.diagnostickeys == [:accept]

@test diagnostics(s) == Dict(:accept => true)

z = BasicDiscUnvParameterState(Int64(-3), [:accept], Float64, [false])
w = deepcopy(z)
@test isequal(z, w)
