##########################################################################
#
#    Model Expression parsing
#      - transforms MCMC specific idioms (~) into regular Julia syntax
#      - calls the ReverseDiffSource module for gradient code generation
#      - creates function
#
##########################################################################

using Base.LinAlg.BLAS  # needed for some fast matrix operations
using ReverseDiffSource

# Distributions extensions (vectorizations on the distribution parameter)
include("definitions/DistributionsExtensions.jl")

# Add new derivation rules to Autodiff for LLAcc type
include("definitions/AccumulatorDerivRules.jl")

# Add new derivation rules to Autodiff for distributions
include("definitions/MCMCDerivRules.jl")

# misc. expression manipulation functions
include("expr_funcs.jl")

# naming conventions
const ACC_SYM = :__acc       # name of accumulator variable
const PARAM_SYM = :__beta    # name of parameter vector


#######################################################################
#   generates the log-likelihood function
#######################################################################
# - 'init' contains the dictionary of model params and their initial value
# - If 'debug' is set to true, the function returns only the function expression
#  that would have been created

function generateModelFunction(model::Expr; gradient=false, debug=false, init...)

	model.head != :block && (model = Expr(:block, model))  # enclose in block if needed
	length(model.args)==0 && error("model should have at least 1 statement")

	vsize, pmap, vinit = modelVars(;init...) # model param info

	model = translate(model) # rewrite ~ statements
	rv = symbol("$(ACC_SYM)v")  # final result in this variable
	model = Expr(:block, [ :($ACC_SYM = LLAcc(0.)), # add log-lik accumulator initialization
		                   model.args, 
		                   # :( $ACC_SYM = $(Expr(:., ACC_SYM, Expr(:quote, :val)) ) )]... )
		                   :( $rv = $(Expr(:., ACC_SYM, Expr(:quote, :val)) ) )]... )

	## build function expression
	if gradient  # case with gradient
		head, body, outsym = ReverseDiffSource.reversediff(model, 
			                                               rv, false, MCMC; 
			                                               init...)

		body = [ vec2var(;init...),  # assigments beta vector -> model parameter vars
		         body.args,
		         :(($outsym, $(var2vec(;init...))))]

		# enclose in a try block
		body = Expr(:try, Expr(:block, body...),
				          :e, 
				          quote 
				          	if isa(e, OutOfSupportError)
				          		return(-Inf, zero($PARAM_SYM))
				          	else
				          		rethrow(e)
				          	end
				          end)

	else  # case without gradient
		head, body, outsym = ReverseDiffSource.reversediff(model, 
			                                               rv, true, MCMC; 
			                                               init...)

		body = [ vec2var(;init...),  # assigments beta vector -> model parameter vars
		         body.args,
		         outsym ]

		# enclose in a try block
		body = Expr(:try, Expr(:block, body...),
				          :e, 
				          quote 
				          	if isa(e, OutOfSupportError)
				          		return(-Inf)
				          	else
				          		rethrow(e)
				          	end
				          end)

	end

	# build and evaluate the let block containing the function and var declarations
	fn = gensym("ll")
	body = Expr(:function, Expr(:call, fn, :($PARAM_SYM::Vector{Float64})),	Expr(:block, body) )
	body = Expr(:let, Expr(:block, :(global $fn), head.args..., body))

	# println("#############\n$body\n############")

	debug ? body : (eval(body) ; (eval(fn), vsize, pmap, vinit) )
end
