#################################################################
#
#    Definition of MCMCModel types 
#
#################################################################

export MCMCLikModel, model

### Model types hierarchy to allow restrictions on applicable samplers
abstract Model
abstract MCMCModel <: Model
# abstract MCMCModelWithGradient <: MCMCModel
# abstract MCMCModelWithHessian <: MCMCModelWithGradient

######### parameters map info  ############
# These types are used to map scalars in the
#   parameter vector to user facing variables
#
immutable PDims
	pos::Integer   # starting position of parameter in the parameter vector
	dims::Tuple    # dimensions of user facing parameter, can be a scalar, vector or matrix
end

typealias PMap Dict{Symbol, PDims}

function ispartition(m::PMap, n::Integer)
	c = zeros(n)
	for v in values(m)
		c[v.pos:(v.pos+prod(v.dims)-1)] += 1
	end
	all(c .== 1.)
end

#### misc functions common to all models  ####
hasvectormethod(f::Function) = !isgeneric(f) | length(methods(f, (Vector{Float64},))) == 1
hasgradient{M<:MCMCModel}(m::M) = m.evalg != nothing
hastensor{M<:MCMCModel}(m::M) = m.evalt != nothing
hasdtensor{M<:MCMCModel}(m::M) = m.evaldt != nothing

#### User-facing model creation function  ####

# TODO

function model(f::Function; mtype="likelihood", args...)
	if mtype == "likelihood"
		return MCMCLikelihoodModel(f; args...)
	elseif mtype == "whatever"
	else
	end
end

function model(f1::Function, f2::Function; mtype="likelihood", args...)
	if mtype == "likelihood"
		return MCMCLikelihoodModel(f1, f2; args...)
	elseif mtype == "whatever"
	else
	end
end

function model(f1::Function, f2::Function, f3::Function; mtype="likelihood", args...)
	if mtype == "likelihood"
		return MCMCLikelihoodModel(f1, f2, f3; args...)
	elseif mtype == "whatever"
	else
	end
end

function model(m::Expr; mtype="likelihood", args...)
	if mtype == "likelihood"
		return MCMCLikelihoodModel(m::Expr; args...)
	elseif mtype == "whatever"
	else
	end
end

#### models  #####

include("likmodel.jl")

# Uncommented temporarily the line below in order to run the examples and debug devel
include("bayesglmmodels.jl")  # TODO : not yet adapted


