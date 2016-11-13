using Base.Test
using Klara

fnames = (
  :value,
  :loglikelihood,
  :logprior,
  :logtarget
)

println("    Testing BasicDiscMuvParameterState constructors and methods...")

v = Int64[-6, 3]
s = BasicDiscMuvParameterState(v)

@test eltype(s) == (Int64, Float64)
@test s.value == v
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == []
@test s.size == length(v)
@test s.diagnostickeys == Symbol[]

@test diagnostics(s) == Dict{Symbol, Any}()

v = Int16[-9, 10, 2]
dv = [false]
s = BasicDiscMuvParameterState(v, [:accept], Float16, dv)

@test eltype(s) == (Int16, Float16)
@test s.value == v
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == dv
@test s.size == length(v)
@test s.diagnostickeys == [:accept]

@test diagnostics(s) == Dict(:accept => dv[1])

ssize = 5
s = BasicDiscMuvParameterState(ssize)

@test eltype(s) == (Int64, Float64)
@test isa(s.value, Vector{Int64})
@test length(s.value) == ssize
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == []
@test s.size == ssize
@test s.diagnostickeys == Symbol[]

@test diagnostics(s) == Dict{Symbol, Any}()

ssize = 2
dv = [true]
s = BasicDiscMuvParameterState(ssize, [:accept], Int32, Float32, dv)

@test eltype(s) == (Int32, Float32)
@test isa(s.value, Vector{Int32})
@test length(s.value) == ssize
for i in 2:4
  @test isnan(getfield(s, fnames[i]))
end
@test s.diagnosticvalues == dv
@test s.size == ssize
@test s.diagnostickeys == [:accept]

@test diagnostics(s) == Dict(:accept => dv[1])

z = BasicDiscMuvParameterState(ones(Int64, 4), [:accept], Float64, [true])
w = deepcopy(z)
@test isequal(z, w)
