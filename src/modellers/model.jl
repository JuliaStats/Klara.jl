#################################################################
#
#    Definition of MCMCModel types 
#
#################################################################

export MCMCModel, MCMCModelWithGradient

######### parameters map info  ############
immutable PDims
	pos::Integer   # starting position of parameter in the parameter vector
	dims::Tuple    # dimensions of user facing parameter, can be a scalar, vector or matrix
end

## Basic (i.e. just a loglik evaluation function) model type
type MCMCModel
	eval::Function                 # log-likelihood evaluation function
	pmap::Dict{Symbol, PDims}      # map to/from parameter vector from/to user-friendly variables
	size::Integer                  # parameter vector size
	init::Vector{Float64}          # parameter vector initial values

	function MCMCModel(f::Function, pmap::Dict{Symbol, PDims} , s::Integer, i::Vector{Float64})
		assert(s>0, "size should be > 0")
		assert(size(i)==(s,), "initial values vector and size not consistent ($(size(i)) <> $s)")

		# check that map is a partition of a vector[1:s]
		c = zeros(s)
		for v in values(pmap)
			c[v.pos:(v.pos+prod(v.dims)-1)] +=1
		end
		assert(all(c .== 1.), "param map is not a partition of parameter vector")

		# check that function can be called with a vector of Float64 as parameter
		assert(length(methods(f, (Vector{Float64},))) == 1, "function cannot be called with Vector{Float64}")

		new(f, pmap, s, i)
	end
end

# Model creation with default values
MCMCModel(f::Function) = MCMCModel(f, 1)
MCMCModel(f::Function, s::Integer) = MCMCModel(f, s, zeros(s))
MCMCModel(f::Function, s::Integer, i::Vector{Float64}) = MCMCModel(f, Dict([:beta], [PDims(1, (s,))]), s, i)


MCMCModel(x-> -dot(x,x))

test = {:a => PDims(3, (2,2)), :b => PDims(7, (10,)), c: => PDims(1, (2)), :d => PDims(17, tuple())}
test = {:a => PDims(3, (2,2)), :b => PDims(7, (10,)), :c => PDims(1, (2,)), :d => PDims(17, tuple())}

c = zeros(20)
v = test[1]
for v in values(test)
	println(v.pos:(v.pos+prod(v.dims)-1))
	c[v.pos:(v.pos+prod(v.dims)-1)] +=1
end
c

f(x::Real) = x
typeof(which(f, zeros(10))) == nothing
f(x::Vector{Float64}) = x

length(methods(f, (Vector{Float64},))) == 1
disassemble(f, (Real,))

#######  model creation from expression stuff + auto diff  ##########
import Base.sum
sum(x::Real) = x  # meant to avoid the annoying behaviour of sum(Inf) 

export generateModelFunction

# naming conventions
const ACC_SYM = :__acc       # name of accumulator variable
const PARAM_SYM = :__beta    # name of parameter vector
const TEMP_NAME = "tmp"      # prefix of temporary variables in log-likelihood function
const DERIV_PREFIX = "d"     # prefix of gradient variables in log-likelihood function

include("modellers/parsing.jl")      #  include model processing functions		
include("modellers/diff.jl")         #  include derivatives definitions
include("modellers/distribs.jl")     #  include distributions definitions
