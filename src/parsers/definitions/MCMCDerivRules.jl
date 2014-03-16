##########################################################################################
#
#    MCMC specific derivation rules
#
##########################################################################################

####### creates multiple rules at once for logpdf(Distrib, x)
# FIXME : using undocumented dprefix function of ReverseDiffSource (should be replaced)
macro dlogpdfd(dist::Symbol, rule)
	sig = :( logpdf($(Expr(:(::), :d, dist)), x::Real) )
	ReverseDiffSource.deriv_rule( sig, :d, rule ) 

	sig = :( logpdf($(Expr(:(::), :d, dist)), x::AbstractArray) )
	rule2 = ReverseDiffSource.substSymbols(rule, {:x => :(x[i]), :ds => :(ds[i])})
	ReverseDiffSource.deriv_rule( sig, :d, :(for i in 1:length(x) ; $rule2 ; end))

	sig = :( logpdf($(Expr(:(::), :d, Expr(:curly, :Array, dist))), x::AbstractArray) )
	rule2 = ReverseDiffSource.substSymbols(rule, {:dd1 => :(dd1[i]), :dd2 => :(dd2[i]), :dd3 => :(dd3[i]), 
		:x => :(x[i]), :ds => :(ds[i]), :d => :(d[i]) })
	ReverseDiffSource.deriv_rule(sig, :d, :(for i in 1:length(x) ; $rule2 ; end))
end

macro dlogpdfx(dist::Symbol, rule)
	sig = :( logpdf($(Expr(:(::), :d, dist)), x::Real) )
	ReverseDiffSource.deriv_rule( sig, :x, rule ) 

	sig = :( logpdf($(Expr(:(::), :d, dist)), x::AbstractArray) )
	rule2 = ReverseDiffSource.substSymbols(rule, {:dx => :(dx[i]), :x => :(x[i]), :ds => :(ds[i])})
	ReverseDiffSource.deriv_rule( sig, :x, :(for i in 1:length(x) ; $rule2 ; end))

	sig = :( logpdf($(Expr(:(::), :d, Expr(:curly, :Array, dist))), x::AbstractArray) )
	rule3 = ReverseDiffSource.substSymbols(rule2, {:d => :(d[i])})
	ReverseDiffSource.deriv_rule( sig, :x, :(for i in 1:length(x) ; $rule3 ; end))
end


####### derivation for Distribution types constructors
ReverseDiffSource.declareType(Distribution, :Distribution)

for d in [:Bernoulli, :TDist, :Exponential, :Poisson]  
	ReverseDiffSource.declareType(eval(d), d)

	ReverseDiffSource.deriv_rule(:( ($d)(p::Real) ), :p, :( dp = ds1 ))
	ReverseDiffSource.deriv_rule(:( ($d)(p::AbstractArray) ), :p, :( copy!(dp, ds1) ))
end

for d in [ :Normal, :Uniform, :Weibull, :Gamma, :Cauchy, :LogNormal, :Binomial, :Beta, :Laplace]
	ReverseDiffSource.declareType(eval(d), d)

	ReverseDiffSource.deriv_rule(:( ($d)(p1::Real, p2::Real) ),   :p1, :( dp1 = ds1 ) )
	ReverseDiffSource.deriv_rule(:( ($d)(p1::Real, p2::Real) ),   :p2, :( dp2 = ds2 ) )
	ReverseDiffSource.deriv_rule(:( ($d)(p1::AbstractArray, p2::AbstractArray) ), :p1, :( copy!(dp1, ds1) ) )
	ReverseDiffSource.deriv_rule(:( ($d)(p1::AbstractArray, p2::AbstractArray) ), :p2, :( copy!(dp2, ds2) ) )
end

#######   Normal distribution
@dlogpdfx Normal dx += (d.μ - x) / (d.σ * d.σ) * ds
@dlogpdfd Normal ( 	dd1 += (x - d.μ) / (d.σ*d.σ) * ds;
					dd2 += ((x - d.μ)*(x - d.μ) / (d.σ*d.σ) - 1.) / d.σ * ds )

## Uniform distribution
@dlogpdfx Uniform dx += 0.
@dlogpdfd Uniform ( dd1 += (d.a <= x <= d.b) / (d.b - d.a) * ds ;
					dd2 += (d.a <= x <= d.b) / (d.a - d.b) * ds )

## Weibull distribution
@dlogpdfd Weibull   ( 	dd1 += ((1. - (x/d.scale)^d.shape) * log(x/d.scale) + 1./d.shape) * ds ;
						dd2 += ((x/d.scale)^d.shape - 1.) * d.shape/d.scale * ds )
@dlogpdfx Weibull   dx += ((1. - (x/d.scale)^d.shape) * d.shape - 1.) / x * ds

## Beta distribution
@dlogpdfd Beta   ( dd1 += (digamma(d.alpha+d.beta) - digamma(d.alpha) + log(x)) * ds ;
				    dd2 += (digamma(d.alpha+d.beta) - digamma(d.beta) + log(1-x)) * ds )
@dlogpdfx Beta   dx += ((d.alpha-1) / x - (d.beta-1)/(1-x)) * ds


## TDist distribution
@dlogpdfd TDist   dd1 += ((x*x-1)/(x*x + d.df)+log(d.df/(x*x+d.df))+digamma((d.df+1)/2)-digamma(d.df/2))/2 * ds
@dlogpdfx TDist   dx += (-(d.df+1)*x / (d.df+x*x)) * ds

## Exponential distribution
@dlogpdfd Exponential   dd1 += (x-d.scale) / (d.scale*d.scale) * ds
@dlogpdfx Exponential   dx -= ds / d.scale

## Gamma distribution
@dlogpdfd Gamma   ( dd1 += (log(x) - log(d.scale) - digamma(d.shape)) * ds ;
					dd2 += ((x - d.scale*d.shape) / (d.scale*d.scale)) * ds )
@dlogpdfx Gamma   dx += (-( d.scale + x - d.shape*d.scale)/(d.scale*x)) * ds

## Cauchy distribution
@dlogpdfd Cauchy   ( dd1 += (2(x-d.location) / (d.scale*d.scale + (x-d.location)*(x-d.location))) * ds ;
					 dd2 += (((x-d.location)*(x-d.location) - d.scale*d.scale) / (d.scale*(d.scale*d.scale + (x-d.location)*(x-d.location)))) * ds )
@dlogpdfx Cauchy   dx += (2(d.location-x) / (d.scale*d.scale + (x-d.location)*(x-d.location))) * ds

## Log-normal distribution
@dlogpdfd LogNormal   ( dd1 += (log(x) - d.meanlog) / (d.sdlog*d.sdlog) * ds ;
					 	dd2 += (d.meanlog*d.meanlog - d.sdlog*d.sdlog - log(x)*(2d.meanlog-log(x))) / (d.sdlog*d.sdlog*d.sdlog) * ds )
@dlogpdfx LogNormal   dx += (d.meanlog - d.sdlog*d.sdlog - log(x)) / (d.sdlog*d.sdlog*x) * ds 

## Laplace distribution
@dlogpdfd Laplace   ( dd1 += (x > d.location ? 1 : -1) / d.scale * ds ;  # location
	                  dd2 += ( abs(x-d.location)/d.scale - 2. ) /d.scale ) # scale
@dlogpdfx Laplace   dx += (x > d.location ? -1 : 1) / d.scale * ds


# Note : vectorization will not be easily possible for multi variate distribs (Dirichlet, Categorical)
# TODO : add other continuous distribs ? : Pareto, Rayleigh, Logistic, Levy, Dirichlet, FDist
# TODO : add other discrete distribs ? : NegativeBinomial, DiscreteUniform, HyperGeometric, Geometric, Categorical

## Bernoulli distribution (Note : no derivation on x parameter as it is an integer)
@dlogpdfd Bernoulli     dd1 += 1. / (d.p1 - 1. + x) * ds

## Binomial distribution (Note : no derivation on x and n parameters as they are integers)
@dlogpdfd Binomial      dd2 += (x / d.prob - (d.size-x) / (1 - d.prob)) * ds

## Poisson distribution (Note : no derivation on x parameter as it is an integer)
@dlogpdfd Poisson       dd1 += (x / d.lambda - 1) * ds


