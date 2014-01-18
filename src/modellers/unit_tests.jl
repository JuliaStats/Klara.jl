## some unit tests for internal functions ##

using Base.Test

module Sandbox 
    include("modelparser.jl")
end

## translate()
@test Sandbox.translate(12)                  == 12
@test Sandbox.translate(:(b = a+6))          == :(b = a+6)
@test Sandbox.translate(:(copy!(b,z)))       == :(copy!(b,z))
@test Sandbox.translate(:(z ~ Normal(1,2)))  == :(__acc += logpdf(Normal(1,2),z))
@test Sandbox.translate([:(b = Array(Float64,10,10)), :(z ~ Normal(1,2))]) == 
	[:(b = Array(Float64,10,10)), :(__acc += logpdf(Normal(1,2),z))]

## vec2var()
@test Sandbox.vec2var(x=3.)                  == [:(x=__beta[1])]
@test Sandbox.vec2var(z=[1., 2, 3])          == [:(z=__beta[1:3])]
@test Sandbox.vec2var(A=[12. 45 0; 18 1 2])  == [:(A=reshape(__beta[1:6],2,3))]
@test Sandbox.vec2var(A=[12. 45 0; 18 1 2], x=3., z=[1., 2, 3]) == 
	[:(A=reshape(__beta[1:6],2,3)), :(x=__beta[7]), :(z=__beta[8:10])]

## var2vec()
@test Sandbox.var2vec(x=3.)                  == :([_dx])
@test Sandbox.var2vec(z=[1., 2, 3])          == :([_dz])
@test Sandbox.var2vec(A=[12. 45 0; 18 1 2])  == :([vec(_dA)])
@test Sandbox.var2vec(z=[1., 2, 3], A=[12. 45 0; 18 1 2], x=3.) == :([_dz, vec(_dA), _dx])


## modelVars()
@test Sandbox.modelVars(x=3.)                    == (1, [:x => (1,())]      , [3.])
@test Sandbox.modelVars(x=3)                     == (1, [:x => (1,())]      , [3.])
@test Sandbox.modelVars(z=[1., 2, 3])            == (3, [:z => (1,(3,))]    , [1.,2,3])
@test Sandbox.modelVars(z=[true, false, false])  == (3, [:z => (1,(3,))]    , [1.,0,0])
@test Sandbox.modelVars(A=[12. 45 0; 18 1 2])    == (6, [:A => (1,(2,3))]   , [12.,18,45,1,0,2])
@test Sandbox.modelVars(z=[1., 2, 3], x=3., A=[12. 45 0; 18 1 2]) == 
	(10, [:A => (5,(2,3)), :z => (1,(3,)), :x => (4,())]   , [1.,2,3,3,12,18,45,1,0,2])


