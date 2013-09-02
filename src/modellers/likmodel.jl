#################################################################
#
#    Definition of Likelihood-type model
#
#   (Basic MCMC model type based on evaluating the log-target)
#
#   Examples of other possible models: MCMCHierarchicalModel, 
#      MCMCGPModel, MCMCKernelModel
#
#################################################################

export MCMCLikModel

type MCMCLikelihoodModel <: MCMCModel
	eval::Function                 # log-likelihood evaluation function
	evalg::Union(Nothing,Function) # gradient vector evaluation function
	evalt::Union(Nothing,Function) # tensor evaluation function
	evaldt::Union(Nothing,Function) # tensor derivative evaluation function
  evalallg::Union(Nothing,Function) # 2-tuple (log-lik, gradient vector) evaluation function
  evalallt::Union(Nothing,Function) # 3-tuple (log-lik, gradient vector, tensor) evaluation function
  evalalldt::Union(Nothing,Function) # 4-tuple (log-lik, gradient vector, tensor, tensor derivative) evaluation function
	pmap::PMap                     # map to/from parameter vector from/to user-friendly variables
	size::Integer                  # parameter vector size
	init::Vector{Float64}          # parameter vector initial values
	scale::Vector{Float64}         # scaling hint on parameters

  MCMCLikelihoodModel(	f::Function, 
									g::Union(Nothing, Function), 
									t::Union(Nothing, Function),
									dt::Union(Nothing, Function),
									i::Vector{Float64}, sc::Vector{Float64}, pmap::PMap) = begin

    s = size(i, 1)
    
		assert(ispartition(pmap, s), "param map is not a partition of parameter vector")
		assert(size(sc,1) == s, "scale parameter size ($(size(sc,1))) different from initial values ($s)")

		# check that likelihood function can be called with a vector of Float64 as argument
		assert(hasvectormethod(f), "likelihood function cannot be called with Vector{Float64}")

		# check that gradient function can be called with a vector of Float64 as argument
		assert(g == nothing || hasvectormethod(g), "gradient function cannot be called with Vector{Float64}")

		# check that tensor function can be called with a vector of Float64 as argument
		assert(t == nothing || hasvectormethod(t), "tensor function cannot be called with Vector{Float64}")

		# check that tensor derivative function can be called with a vector of Float64 as argument
		assert(dt == nothing || hasvectormethod(dt), "tensor derivative function cannot be called with Vector{Float64}")

		# check that initial values are in the support of likelihood function
		assert(isfinite(f(i)), "Initial values out of model support, try other values")

    instance = new()
    instance.eval = f
    instance.evalg = g
    instance.evalt = t
    instance.evaldt = dt
    instance.init = i
    instance.scale = sc
    instance.pmap =pmap
    
    instance.size = s
    instance.evalallg = (pars::Vector{Float64} -> (instance.eval(pars), instance.evalg(pars)))
    instance.evalallt = (pars::Vector{Float64} -> (instance.eval(pars), instance.evalg(pars), instance.evalt(pars)))
    instance.evalalldt = (pars::Vector{Float64} -> (instance.eval(pars), instance.evalg(pars), instance.evalt(pars), instance.evaldt(pars)))
        
		instance
	end
end

typealias MCMCLikModel MCMCLikelihoodModel

# Model creation : gradient or tensor or tensor derivative not specified
MCMCLikelihoodModel( lik::Function; args...) = 
	MCMCLikelihoodModel(lik, nothing, nothing, nothing; args...)
MCMCLikelihoodModel( lik::Function, grad::Function; args...) = 
	MCMCLikelihoodModel(lik, grad, nothing, nothing; args...)
MCMCLikelihoodModel( lik::Function, grad::Function, tensor::Function; args...) = 
	MCMCLikelihoodModel(lik, grad, tensor, nothing; args...)

# Model creation : gradient+tensor+tensor derivative version 
function MCMCLikelihoodModel(	lik::Function, 
								grad::Union(Nothing, Function), 
								tensor::Union(Nothing, Function),
								dtensor::Union(Nothing, Function); 
								init::Union(Real, Vector{Float64}) = [1.0], 
								scale::Union(Real, Vector{Float64}) = 1.0,
								pmap::Union(Nothing, PMap) = nothing) 

	# convert init to vector if needed
	init = isa(init, Real) ? [init] : init

	# expand scale to parameter vector size if needed
	scale = isa(scale, Real) ? scale * ones(length(init)) : scale

	# all parameters named "pars" by default
	pmap = pmap == nothing ? Dict([:pars], [PDims(1, size(init))]) : pmap 

	MCMCLikelihoodModel(lik, grad, tensor, dtensor, init, scale, pmap)
end

# Model creation using expression parsing and autodiff
function MCMCLikelihoodModel(	m::Expr; 
								gradient::Bool=false,
								init=nothing,
								pmap=nothing,
								scale::Union(Real, Vector{Float64}) = 1.0,
								args...)

	# when using expressions, initial values are passed in keyword args
	#  with one arg by parameter, therefore there is not need for an init arg
	assert(init == nothing, "'init' kwargs not allowed for model as expression\n")

	# same thing with 'pmap'
	assert(pmap == nothing, "'pmap' kwargs not allowed for model as expression\n")

	# generate lik function
	f, s, p, i = generateModelFunction(m; gradient=false, args...) # loglik only function

	# generate gradient function if requested
	if gradient
		g, s, p, i = generateModelFunction(m; gradient=true, args...) # loglik and gradient function
	else
		g = nothing
	end

	MCMCLikelihoodModel(f, g, nothing, nothing; init=i, pmap=p, scale=scale)
end
