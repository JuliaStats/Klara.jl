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

# The likelihood Model type
type MCLikelihood <: MCModel
	eval::Function              # log-likelihood evaluation function
	evalg::FunctionOrNothing 			# gradient vector evaluation function
	evalt::FunctionOrNothing 			# tensor evaluation function
	evaldt::FunctionOrNothing 			# tensor derivative evaluation function
  evalallg::FunctionOrNothing 		# 2-tuple (log-lik, gradient vector) evaluation function
  evalallt::FunctionOrNothing 		# 3-tuple (log-lik, gradient vector, tensor) evaluation function
  evalalldt::FunctionOrNothing 		# 4-tuple (log-lik, gradient vector, tensor, tensor derivative) evaluation function
	pmap::Dict                  # map to/from parameter vector from/to user-friendly variables
	size::Int                   # parameter vector size
	init::Vector{Float64}       # parameter vector initial values
	scale::Vector{Float64}      # scaling hint on parameters

	MCLikelihood(f::Function, 
						g::FunctionOrNothing, ag::FunctionOrNothing,
						t::FunctionOrNothing, at::FunctionOrNothing,
						dt::FunctionOrNothing, adt::FunctionOrNothing,
						init::Vector{Float64}, 
						sc::Vector{Float64}, 
						pmap::Dict) = begin

		s = size(init, 1)

		@assert ispartition(pmap, s) "param map is not a partition of parameter vector"
		@assert size(sc,1) == s "scale parameter size ($(size(sc,1))) different from initial values ($s)"

    fin = [f, g, ag, t, at, dt, adt]
    fout = Array(FunctionOrNothing, 7)

		# check that all functions can be called with a vector of Float64 as argument
		for i = 1:7
			if isgeneric(fin[i]) && !method_exists(fin[i], (Vector{Float64},))
				if method_exists(fin[i], (Float64,))
					fout[i] = x::Vector{Float64} -> fin[i](x[1])  
				else
				  error("one of the supplied functions cannot be called with Vector{Float64}")
          #TODO : make error message print which function is problematic
        end
			else
			    fout[i] = fin[i]
			end
		end

		# check that initial values are in the support of likelihood function
		@assert isfinite(fout[1](init)) "Initial values out of model support, try other values"

    new(fout[1], fout[2], fout[4], fout[6], fout[3], fout[5], fout[7], pmap, s, init, sc)
	end
end

function show(io::IO, res::MCLikelihood)
  print(io, "LikelihoodModel, with $(length(res.pmap)) parameter(s)")
  hasgradient(res) && print(io, ", with gradient")
  hastensor(res) && print(io, "/tensor")
  hasdtensor(res) && print(io, "/dtensor")
  println(io)
end

# Model creation using expression parsing and autodiff
function MCLikelihood(	m::Expr; 
								gradient::Bool=false,
								init=nothing,
								pmap=nothing,
								scale::F64OrVectorF64 = 1.0,
								args...)
	# when using expressions, initial values are passed in keyword args
	#  with one arg by parameter, therefore there is not need for an init arg
	@assert init == nothing "'init' kwargs not allowed for model as expression\n"

	# same thing with 'pmap'
	@assert pmap == nothing "'pmap' kwargs not allowed for model as expression\n"

	# generate lik function
	f, s, p, i = generateModelFunction(m; gradient=false, args...) # loglik only function

	# generate gradient function if requested
	if gradient
		g, s, p, i = generateModelFunction(m; gradient=true, args...) # loglik and gradient function
	else
		g = nothing
	end

	MCLikelihood(f, allgrad=g, init=i, pmap=p, scale=scale)
end


# Model creation : with user supplied functions
function MCLikelihood(	lik::Function;
								grad::FunctionOrNothing = nothing, 
								tensor::FunctionOrNothing = nothing,
								dtensor::FunctionOrNothing = nothing,
								allgrad::FunctionOrNothing = nothing, 
								alltensor::FunctionOrNothing = nothing,
								alldtensor::FunctionOrNothing = nothing,
								init::F64OrVectorF64 = [1.0], 
								scale::F64OrVectorF64 = 1.0,
								pmap::Union(Nothing, Dict) = nothing) 

	# convert init to vector if needed
	init = isa(init, Float64) ? [init] : init

	# expand scale to parameter vector size if needed
	scale = isa(scale, Float64) ? scale * ones(length(init)) : scale

	# all parameters named "pars" by default
	if pmap == nothing ; pmap = Dict(zip([:pars], [(1, size(init))])) ; end 

	# now build missing functions, if any
	fmat = Array(FunctionOrNothing, 3, 2)
	for (i, f1, f2) in [(1, grad, allgrad), 
						(2, tensor, alltensor), 
						(3, dtensor, alldtensor)]
		fmat[i,:] = [f1 f2]
		if f1==nothing && f2!=nothing # only the tuple version is supplied
			fmat[i,1] = (v) -> f2(v)[end] 
		elseif f1!=nothing && f2==nothing # only the single version is supplied
			if i == 1
				fmat[i,2] = (v) -> (lik(v), f1(v))
			else
				@assert isa(fmat[i-1,2], Function) "missing function !"
				fmat[i,2] = (v) -> tuple(fmat[i-1,2](v)..., f1(v))
			end
		end
	end

	MCLikelihood(lik, 
						fmat[1,1], fmat[1,2],
						fmat[2,1], fmat[2,2],
						fmat[3,1], fmat[3,2],
						init, scale, pmap)
end

# Model creation with multivariate Distribution as input

function MCLikelihood(d::MultivariateDistribution;
  grad::FunctionOrNothing = nothing, 
  tensor::FunctionOrNothing = nothing,
  dtensor::FunctionOrNothing = nothing,
  allgrad::FunctionOrNothing = nothing, 
  alltensor::FunctionOrNothing = nothing,
  alldtensor::FunctionOrNothing = nothing,
	init::F64OrVectorF64 = [1.0], scale::F64OrVectorF64 = 1.0, pmap::Union(Nothing, Dict) = nothing)
  @assert method_exists(logpdf, (typeof(d), Vector{Float64})) "logpdf function not defined for $d"

  fout = Array(FunctionOrNothing, 2)
  fout[1] = x::Vector{Float64} -> logpdf(d, x)

  if grad == nothing
  	if method_exists(gradlogpdf, (typeof(d), Vector{Float64}))
  		fout[2] = x::Vector{Float64} -> gradlogpdf(d, x)
  	end
  end

	MCLikelihood(fout[1]; grad=fout[2], tensor=tensor, dtensor=dtensor,
    allgrad=allgrad, alltensor=alltensor, alldtensor=alldtensor,
    init=init, scale=scale, pmap=pmap)
end

# Model creation with univariate Distribution as input

function MCLikelihood(d::UnivariateDistribution;
  grad::FunctionOrNothing = nothing, 
  tensor::FunctionOrNothing = nothing,
  dtensor::FunctionOrNothing = nothing,
  allgrad::FunctionOrNothing = nothing, 
  alltensor::FunctionOrNothing = nothing,
  alldtensor::FunctionOrNothing = nothing,
	init::F64OrVectorF64 = [1.0], scale::F64OrVectorF64 = 1.0, pmap::Union(Nothing, Dict) = nothing)
  @assert method_exists(logpdf, (typeof(d), Float64)) "logpdf function not defined for $d"

  fout = Array(FunctionOrNothing, 2)
  fout[1] = x::Vector{Float64} -> logpdf(d, x[1])

  if grad == nothing
  	if method_exists(gradlogpdf, (typeof(d), Float64))
  		fout[2] = x::Vector{Float64} -> [gradlogpdf(d, x[1])]
  	end
  end

	MCLikelihood(fout[1]; grad=fout[2], tensor=tensor, dtensor=dtensor,
    allgrad=allgrad, alltensor=alltensor, alldtensor=alldtensor,
    init=init, scale=scale, pmap=pmap)
end
