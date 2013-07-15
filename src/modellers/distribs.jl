##########################################################################################
#
#    Definition of distributions
#
##########################################################################################

########## locally defined distributions #############
# defined within the library, either because libRmath doesn't have it (Bernoulli)
#  or because there is a significant speed gain
# TODO : a few other distributions should gain speed also, should be tested

function logpdfBernoulli(prob::Real, x::Real)
	assert(0. <= prob <= 1., "calling Bernoulli with prob > 1. or < 0.")
	if x == 0.
		prob == 1. ? throw("give up eval") : return(log(1. - prob))
	elseif x == 1.
		prob == 0. ? throw("give up eval") : return(log(prob))
	elseif
	 	error("calling Bernoulli with variable other than 0 or 1 (false or true)")
	end
end

function logpdfNormal(mu::Real, sigma::Real, x::Real)
	local const fac = -log(sqrt(2pi))
	assert(sigma > 0., "calling logpdfNormal with negative or null stdev")
	local r = (x-mu)/sigma
	return -r*r*0.5-log(sigma)+fac
end

function logpdfUniform(a::Real, b::Real, x::Real)
	assert(a < b, "calling logpdfUniform with lower limit >= upper limit")
	return (a <= x <= b) ? -log(b-a) : throw("give up eval")
end

########### distributions using libRmath ######### 

_jl_libRmath = dlopen("libRmath")

#          Name           libRmath Name    Arity       !!! empty lib name means local def
dists = { (:Weibull, 	  "dweibull",       3),
		  (:Binomial, 	  "dbinom",         3),
		  (:Gamma,  	  "dgamma",         3),
		  (:Cauchy,  	  "dcauchy",        3),
		  (:logNormal,    "dlnorm",         3),
		  (:Beta, 	      "dbeta",          3),
		  (:Poisson,  	  "dpois",          2),
		  (:TDist,  	  "dt",             2),
		  (:Exponential,  "dexp",           2),
		  (:Uniform, 	  "",               3),     #  dunif in libRmath
	      (:Normal,  	  "",               3),     #  "dnorm4" in libRmath
		  (:Bernoulli,    "",               2)}


for d in dists  # d = dists[3]
	if d[2] != "" # empty means locally defined
		fsym = symbol("logpdf$(d[1])")

		npars = d[3]
		args = [symbol("x$i") for i in [npars, 1:(npars-1)...]] # in correct order for libRmath

		fex = quote
			function ($fsym)($([Expr(:(::), symbol("x$i"), :Real) for i in 1:npars]...))
				local res = 0.

		        res = ccall(dlsym(_jl_libRmath, $(string(d[2]))), Float64,
		            	 	  $(Expr(:tuple, [[:Float64 for i in 1:npars]..., :Int32 ]...)),
		            	 	  $(args...), 1	)

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
end

####### all possible vectorized versions  ##########

for d in dists # d = dists[1]
	fsym = symbol("logpdf$(d[1])")

	arity = d[3]
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


