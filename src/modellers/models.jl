#################################################################
#
#    Definition of MCMCModel types 
#
#################################################################

export Model, ModelG

### Model types hierarchy to allow restrictions applicable samplers
abstract MCMCModel
abstract MCMCModelWithGradient <: MCMCModel
abstract MCMCModelWithHessian <: MCMCModelWithGradient

######### parameters map info  ############
immutable PDims
	pos::Integer   # starting position of parameter in the parameter vector
	dims::Tuple    # dimensions of user facing parameter, can be a scalar, vector or matrix
end

######### Basic (i.e. just a loglik evaluation function) model type ############
type Model <: MCMCModel
	eval::Function                 # log-likelihood evaluation function
	pmap::Dict{Symbol, PDims}      # map to/from parameter vector from/to user-friendly variables
	size::Integer                  # parameter vector size
	init::Vector{Float64}          # parameter vector initial values

	function Model(f::Function, pmap::Dict{Symbol, PDims} , s::Integer, i::Vector{Float64})
		assert(s>0, "size should be > 0")
		assert(size(i)==(s,), "initial values vector and size not consistent ($(size(i)) <> $s)")

		# check that map is a partition of a vector[1:s]
		c = zeros(s)
		for v in values(pmap)
			c[v.pos:(v.pos+prod(v.dims)-1)] +=1
		end
		assert(all(c .== 1.), "param map is not a partition of parameter vector")

		# check that function can be called with a vector of Float64 as parameter
		assert(!isgeneric(f) | length(methods(f, (Vector{Float64},))) == 1, 
			"function cannot be called with Vector{Float64}")

		new(f, pmap, s, i)
	end
end

# Model creation with default values
Model(f::Function) = Model(f, 1)
Model(f::Function, s::Integer) = Model(f, s, zeros(s))
Model(f::Function, s::Integer, i::Vector{Float64}) = Model(f, Dict([:beta], [PDims(1, (s,))]), s, i)

# ModelG creation using expression parsing and autodiff
function Model(m::Expr; init...)
	f, s, p, i = generateModelFunction(m; gradient=false, init...)  # loglik only function

	Model(f, p, s, i)
end





########### Model with gradient function  ####################
type ModelG <: MCMCModelWithGradient
	eval::Function                 # log-likelihood evaluation function
	evalg::Function                # tuple (log-lik, gradient vector) evaluation function
	pmap::Dict{Symbol, PDims}      # map to/from parameter vector from/to user-friendly variables
	size::Integer                  # parameter vector size
	init::Vector{Float64}          # parameter vector initial values

	function ModelG(f::Function, g::Function, pmap::Dict{Symbol, PDims}, s::Integer, i::Vector{Float64})
		assert(s>0, "size should be > 0")
		assert(size(i)==(s,), "initial values vector and size not consistent ($(size(i)) <> $s)")

		# check that map is a partition of a vector[1:s]
		c = zeros(s)
		for v in values(pmap)
			c[v.pos:(v.pos+prod(v.dims)-1)] +=1
		end
		assert(all(c .== 1.), "param map is not a partition of parameter vector")

		# check that function can be called with a vector of Float64 as parameter
		assert(!isgeneric(f) | length(methods(f, (Vector{Float64},))) == 1, 
			"function cannot be called with Vector{Float64}")
		assert(!isgeneric(g) | length(methods(g, (Vector{Float64},))) == 1, 
			"function cannot be called with Vector{Float64}")

		new(f, g, pmap, s, i)
	end
end

# Model creation with default values
ModelG(f::Function, g::Function) = ModelG(f, g, 1)
ModelG(f::Function, g::Function, s::Integer) = ModelG(f, g, s, zeros(s))
ModelG(f::Function, g::Function, s::Integer, i::Vector{Float64}) = 
	ModelG(f, g, Dict([:beta], [PDims(1, (s,))]), s, i)

# ModelG creation using expression parsing and autodiff
function ModelG(m::Expr; init...)
	f, s, p, i = generateModelFunction(m; gradient=false, init...)  # loglik only function
	g, s, p, i = generateModelFunction(m; gradient=true, init...)   # loglik and gradient function

	ModelG(f, g, p, s, i)
end

