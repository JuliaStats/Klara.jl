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
	evalg::Union(Nothing,Function) # 2-tuple (log-lik, gradient vector) evaluation function
	evalh::Union(Nothing,Function) # 3-tuple (log-lik, gradient vector, hessian) evaluation function
	evalt::Union(Nothing,Function) # 4-tuple (log-lik, gradient vector, hessian, tensor) evaluation function
	pmap::PMap                     # map to/from parameter vector from/to user-friendly variables
	size::Integer                  # parameter vector size
	init::Vector{Float64}          # parameter vector initial values
	scale::Vector{Float64}         # scaling hint on parameters

	function MCMCLikelihoodModel(	f::Function, 
									g::Union(Nothing, Function), 
									h::Union(Nothing, Function),
									t::Union(Nothing, Function),
									i::Vector{Float64}, sc::Vector{Float64}, pmap::PMap)
		s = size(i,1)

		assert(ispartition(pmap, s), "param map is not a partition of parameter vector")
		assert(size(sc,1) == s, "scale parameter size ($(size(sc,1))) different from initial values ($s)")

		# check that likelihood function can be called with a vector of Float64 as argument
		assert(hasvectormethod(f), "likelihood function cannot be called with Vector{Float64}")

		# check that gradient function can be called with a vector of Float64 as argument
		assert(g == nothing || hasvectormethod(g), "gradient function cannot be called with Vector{Float64}")

		# check that hessian function can be called with a vector of Float64 as argument
		assert(h == nothing || hasvectormethod(h), "hessian function cannot be called with Vector{Float64}")

		# check that hessian function can be called with a vector of Float64 as argument
		assert(t == nothing || hasvectormethod(t), "tensor function cannot be called with Vector{Float64}")

		# check that initial values are in the support of likelihood function
		assert(isfinite(f(i)), "Initial values out of model support, try other values")

		new(f, g, h, t, pmap, s, i, sc)
	end
end

typealias MCMCLikModel MCMCLikelihoodModel

# Model creation : gradient or hessian or tensor not specified
MCMCLikelihoodModel( lik::Function; args...) = 
	MCMCLikelihoodModel(lik, nothing, nothing, nothing; args...)
MCMCLikelihoodModel( lik::Function, grad::Function; args...) = 
	MCMCLikelihoodModel(lik, grad, nothing, nothing; args...)
MCMCLikelihoodModel( lik::Function, grad::Function, hessian::Function; args...) = 
	MCMCLikelihoodModel(lik, grad, hessian, nothing; args...)

# Model creation : gradient+hessian+tensor version 
function MCMCLikelihoodModel(	lik::Function, 
								grad::Union(Nothing, Function), 
								hessian::Union(Nothing, Function),
								tensor::Union(Nothing, Function); 
								init::Union(Real, Vector{Float64}) = [1.0], 
								scale::Union(Real, Vector{Float64}) = 1.0,
								pmap::Union(Nothing, PMap) = nothing) 

	# convert init to vector if needed
	init = isa(init, Real) ? [init] : init

	# expand scale to parameter vector size if needed
	scale = isa(scale, Real) ? scale * ones(length(init)) : scale

	# all parameters named "pars" by default
	pmap = pmap == nothing ? Dict([:pars], [PDims(1, size(init))]) : pmap 

	MCMCLikelihoodModel(lik, grad, hessian, tensor, init, scale, pmap)
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



