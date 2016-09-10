using Base.Test
using Klara

println("    Testing BasicUnvVariableState constructors...")

v = Float64(1.21)
s = BasicUnvVariableState(v)

@test eltype(s) == Float64
@test s.value == v

println("    Testing BasicMuvVariableState constructors...")

v = Float32[1.5, 4.1]
s = BasicMuvVariableState(v)

@test eltype(s) == Float32
@test s.value == v
@test s.size == length(v)

ssize = 3
s = BasicMuvVariableState(ssize, Float16)

@test eltype(s) == Float16
@test s.size == ssize

println("    Testing BasicMavVariableState constructors...")

v = BigFloat[3.11 7.34; 9.7 6.72; 1.18 8.1]
s = BasicMavVariableState(v)

@test eltype(s) == BigFloat
@test s.value == v
@test s.size == size(v)

ssize = (3, 5)
s = BasicMavVariableState(ssize, Float16)

@test eltype(s) == Float16
@test s.size == ssize
