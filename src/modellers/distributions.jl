##########################################################################################
#
#    Definition of distributions
#
#    links logpdfXXXX() functions to logpdf(XXXX(), x) in Distributions.jl
#
##########################################################################################

using Distributions

########## locally defined distributions #############
# defined here because there is a speed gain over using Distributions.jl
# TODO : file an issue in Distributions.jl

# function logpdfNormal(mu::Real, sigma::Real, x::Real)
# 	local const fac = -log(sqrt(2pi))
# 	assert(sigma > 0., "calling logpdfNormal with negative or null stdev")
# 	local r = (x-mu)/sigma
# 	return -r*r*0.5-log(sigma)+fac
# end

# function logpdfUniform(a::Real, b::Real, x::Real)
# 	assert(a < b, "calling logpdfUniform with lower limit >= upper limit")
# 	return (a <= x <= b) ? -log(b-a) : throw("give up eval")
# end

########### distributions using libRmath ######### 

#          Name           Arity
dists = { (:Weibull, 	  3),
		  (:Binomial, 	  3),
		  (:Gamma,  	  3),
		  (:Cauchy,  	  3),
		  (:LogNormal,    3),
		  (:Beta, 	      3),
		  (:Poisson,  	  2),
		  (:TDist,  	  2),
		  (:Exponential,  2),
		  (:Uniform, 	  3),   
	      (:Normal,  	  3),
		  (:Bernoulli,    2)}

for d in dists  # d = dists[3]
	fsym = symbol("logpdf$(d[1])")

	npars = d[2]
	args = [symbol("x$i") for i in [npars, 1:(npars-1)...]] # in correct order for libRmath

	fex = quote
		function ($fsym)($([Expr(:(::), symbol("x$i"), :Real) for i in 1:npars]...))
			local res = 0.

	        res = logpdf( ($(d[1]))($([symbol("x$i") for i in 1:(npars-1)]...)), $(symbol("x$npars")) )

			if res == -Inf
				throw("give up eval")
			elseif isnan(res)
				local ar = $(Expr(:tuple, [args..., 1]))
				error(string("calling ", $(string(fsym)), ar, " returned an error"))
			end
			res 
		end
	end

	eval(fex)
end

####### all possible vectorized versions  ##########

for d in dists # d = dists[1]
	fsym = symbol("logpdf$(d[1])")

	arity = d[2]
	ps = [ ifloor((i-1) / 2^(j-1)) % 2 for i=2:2^arity, j=1:arity]

	for l in 1:size(ps,1) # l = 3
		pars = Symbol[ps[l,j]==0 ? :Real : :AbstractArray for j in 1:arity]

		rv = symbol("x$(findfirst(pars .== :AbstractArray))")
		npars = length(pars)

		fex = quote
			function ($fsym)($([Expr(:(::), symbol("x$i"), pars[i]) for i in 1:npars]...)) 
		        local res = 0.
		        for i in 1:length($rv)
		        	res += ($fsym)($([pars[i]==:Real ? symbol("x$i") : Expr(:ref, symbol("x$i"), :i) for i in 1:npars]...))
		        end
		        res
		    end
		end

		eval(fex)
	end
end


