##########################################################################################
#
#    function 'derive' returning the expr of gradient
#
#
##########################################################################################
# TODO : add operators : hcat, vcat, ? : , map, mapreduce, if else 

## macro to simplify derivation rules creation
macro dfunc(func::Expr, dv::Symbol, diff::Expr) 
	argsn = map(e-> isa(e, Symbol) ? e : e.args[1], func.args[2:end])
	index = find(dv .== argsn)[1]

	# change var names in signature and diff expr to x1, x2, x3, ..
	smap = { argsn[i] => symbol("x$i") for i in 1:length(argsn) }
	args2 = substSymbols(func.args[2:end], smap)

	# diff function name
	fn = symbol("d_$(func.args[1])_x$index")

	fullf = Expr(:(=), Expr(:call, fn, args2...), Expr(:quote, substSymbols(diff, smap)) )
	eval(fullf)
end


## common operators
# TODO : check if ds can be removed for clarity

@dfunc +(x::Real, y)     x     sum(ds)
@dfunc +(x::Array, y)    x     +ds
@dfunc +(x, y::Real)     y     sum(ds)
@dfunc +(x, y::Array)    y     +ds

@dfunc -x                x     -ds
@dfunc -(x::Real, y)     x     sum(ds)
@dfunc -(x::Array, y)    x     +ds
@dfunc -(x, y::Real)     y     -sum(ds)
@dfunc -(x, y::Array)    y     -ds

@dfunc sum(x)       x     +ds
@dfunc dot(x, y)    x     y .* ds
@dfunc dot(x, y)    y     x .* ds

@dfunc log(x)       x     ds ./ x
@dfunc exp(x)       x     exp(x) .* ds

@dfunc sin(x)       x     cos(x) .* ds
@dfunc cos(x)       x     -sin(x) .* ds

@dfunc abs(x)       x     sign(x) .* ds

@dfunc *(x::Real, y)     x     sum(ds .* y)
@dfunc *(x::Array, y)    x     ds * transpose(y)
@dfunc *(x, y::Real)     y     sum(ds .* x)
@dfunc *(x, y::Array)    y     transpose(x) * ds

@dfunc .*(x::Real, y)    x     sum(ds .* y)
@dfunc .*(x::Array, y)   x     ds .* y
@dfunc .*(x, y::Real)    y     sum(ds .* x)
@dfunc .*(x, y::Array)   y     ds .* x

@dfunc ^(x::Real, y::Real)  x     y * x ^ (y-1) * ds # Both args reals
@dfunc ^(x::Real, y::Real)  y     log(x) * x ^ y * ds # Both args reals

@dfunc .^(x::Real, y)    x     sum(y .* x .^ (y-1) .* ds)
@dfunc .^(x::Array, y)   x     y .* x .^ (y-1) .* ds
@dfunc .^(x, y::Real)    y     sum(log(x) .* x .^ y .* ds)
@dfunc .^(x, y::Array)   y     log(x) .* x .^ y .* ds

@dfunc /(x::Real, y)          x     sum(ds ./ y)
@dfunc /(x::Array, y::Real)   x     ds ./ y
@dfunc /(x, y::Real)          y     sum(- x ./ (y .* y) .* ds)
@dfunc /(x::Real, y::Array)   y     (- x ./ (y .* y)) .* ds

@dfunc ./(x::Real, y)        x     sum(ds ./ y)
@dfunc ./(x::Array, y)       x     ds ./ y
@dfunc ./(x, y::Real)        y     sum(- x ./ (y .* y) .* ds)
@dfunc ./(x, y::Array)       y     (- x ./ (y .* y)) .* ds

@dfunc max(x::Real, y)    x     sum((x .> y) .* ds)
@dfunc max(x::Array, y)   x     (x .> y) .* ds
@dfunc max(x, y::Real)    y     sum((x .< y) .* ds)
@dfunc max(x, y::Array)   y     (x .< y) .* ds

@dfunc min(x::Real, y)    x     sum((x .< y) .* ds)
@dfunc min(x::Array, y)   x     (x .< y) .* ds
@dfunc min(x, y::Real)    y     sum((x .> y) .* ds)
@dfunc min(x, y::Array)   y     (x .> y) .* ds

@dfunc transpose(x::Real)   x   +ds
@dfunc transpose(x::Array)  x   transpose(ds)

## Normal distribution
@dfunc logpdfNormal(mu::Real, sigma, x)    mu     sum((x - mu) ./ (sigma .* sigma)) * ds
@dfunc logpdfNormal(mu::Array, sigma, x)   mu     (x - mu) ./ (sigma .* sigma) * ds
@dfunc logpdfNormal(mu, sigma::Real, x)    sigma  sum(((x - mu).*(x - mu) ./ (sigma.*sigma) - 1.) ./ sigma) * ds
@dfunc logpdfNormal(mu, sigma::Array, x)   sigma  ((x - mu).*(x - mu) ./ (sigma.*sigma) - 1.) ./ sigma * ds
@dfunc logpdfNormal(mu, sigma, x::Real)    x      sum((mu - x) ./ (sigma .* sigma)) * ds
@dfunc logpdfNormal(mu, sigma, x::Array)   x      (mu - x) ./ (sigma .* sigma) * ds

## Uniform distribution
@dfunc logpdfUniform(a::Real, b, x)      a   sum((a .<= x .<= b) ./ (b - a)) * ds
@dfunc logpdfUniform(a::Array, b, x)     a   ((a .<= x .<= b) ./ (b - a)) * ds
@dfunc logpdfUniform(a, b::Real, x)      b   sum((a .<= x .<= b) ./ (a - b)) * ds
@dfunc logpdfUniform(a, b::Array, x)     b   ((a .<= x .<= b) ./ (a - b)) * ds
@dfunc logpdfUniform(a, b, x)            x   zero(x)

## Weibull distribution
@dfunc logpdfWeibull(sh::Real, sc, x)    sh  (r = x./sc ; sum(((1. - r.^sh) .* log(r) + 1./sh)) * ds)
@dfunc logpdfWeibull(sh::Array, sc, x)   sh  (r = x./sc ; ((1. - r.^sh) .* log(r) + 1./sh) * ds)
@dfunc logpdfWeibull(sh, sc::Real, x)    sc  sum(((x./sc).^sh - 1.) .* sh./sc) * ds
@dfunc logpdfWeibull(sh, sc::Array, x)   sc  ((x./sc).^sh - 1.) .* sh./sc * ds
@dfunc logpdfWeibull(sh, sc, x::Real)    x   sum(((1. - (x./sc).^sh) .* sh - 1.) ./ x) * ds
@dfunc logpdfWeibull(sh, sc, x::Array)   x   ((1. - (x./sc).^sh) .* sh - 1.) ./ x * ds

## Beta distribution
@dfunc logpdfBeta(a, b, x::Real)      x     sum((a-1) ./ x - (b-1) ./ (1-x)) * ds
@dfunc logpdfBeta(a, b, x::Array)     x     ((a-1) ./ x - (b-1) ./ (1-x)) * ds
@dfunc logpdfBeta(a::Real, b, x)      a     sum(digamma(a+b) - digamma(a) + log(x)) * ds
@dfunc logpdfBeta(a::Array, b, x)     a     (digamma(a+b) - digamma(a) + log(x)) * ds
@dfunc logpdfBeta(a, b::Real, x)      b     sum(digamma(a+b) - digamma(b) + log(1-x)) * ds
@dfunc logpdfBeta(a, b::Array, x)     b     (digamma(a+b) - digamma(b) + log(1-x)) * ds

## TDist distribution
@dfunc logpdfTDist(df, x::Real)     x     sum(-(df+1).*x ./ (df+x.*x)) .* ds
@dfunc logpdfTDist(df, x::Array)    x     (-(df+1).*x ./ (df+x.*x)) .* ds
@dfunc logpdfTDist(df::Real, x)     df    (tmp2 = (x.*x + df) ; sum( (x.*x-1)./tmp2 + log(df./tmp2) + digamma((df+1)/2) - digamma(df/2) ) / 2 .* ds )
@dfunc logpdfTDist(df::Array, x)    df    (tmp2 = (x.*x + df) ; ( (x.*x-1)./tmp2 + log(df./tmp2) + digamma((df+1)/2) - digamma(df/2) ) / 2 .* ds )

## Exponential distribution
@dfunc logpdfExponential(sc, x::Real)    x   sum(-1/sc) .* ds
@dfunc logpdfExponential(sc, x::Array)   x   (- ds ./ sc)
@dfunc logpdfExponential(sc::Real, x)    sc  sum((x-sc)./(sc.*sc)) .* ds
@dfunc logpdfExponential(sc::Array, x)   sc  (x-sc) ./ (sc.*sc) .* ds

## Gamma distribution
@dfunc logpdfGamma(sh, sc, x::Real)    x   sum(-( sc + x - sh.*sc)./(sc.*x)) .* ds
@dfunc logpdfGamma(sh, sc, x::Array)   x   (-( sc + x - sh.*sc)./(sc.*x)) .* ds
@dfunc logpdfGamma(sh::Real, sc, x)    sh  sum(log(x) - log(sc) - digamma(sh)) .* ds
@dfunc logpdfGamma(sh::Array, sc, x)   sh  (log(x) - log(sc) - digamma(sh)) .* ds
@dfunc logpdfGamma(sh, sc::Real, x)    sc  sum((x - sc.*sh) ./ (sc.*sc)) .* ds
@dfunc logpdfGamma(sh, sc::Array, x)   sc  ((x - sc.*sh) ./ (sc.*sc)) .* ds

## Cauchy distribution
@dfunc logpdfCauchy(mu, sc, x::Real)    x   sum(2(mu-x) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu, sc, x::Array)   x   (2(mu-x) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu::Real, sc, x)    mu  sum(2(x-mu) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu::Array, sc, x)   mu  (2(x-mu) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu, sc::Real, x)    sc  sum(((x-mu).*(x-mu) - sc.*sc) ./ (sc.*(sc.*sc + (x-mu).*(x-mu)))) .* ds
@dfunc logpdfCauchy(mu, sc::Array, x)   sc  (((x-mu).*(x-mu) - sc.*sc) ./ (sc.*(sc.*sc + (x-mu).*(x-mu)))) .* ds

## Log-normal distribution
@dfunc logpdflogNormal(lmu, lsc, x::Real)   x    ( tmp2=lsc.*lsc ; sum( (lmu - tmp2 - log(x)) ./ (tmp2.*x) ) .* ds )
@dfunc logpdflogNormal(lmu, lsc, x::Array)  x    ( tmp2=lsc.*lsc ; ( (lmu - tmp2 - log(x)) ./ (tmp2.*x) ) .* ds )
@dfunc logpdflogNormal(lmu::Real, lsc, x)   lmu  sum((log(x) - lmu) ./ (lsc .* lsc)) .* ds
@dfunc logpdflogNormal(lmu::Array, lsc, x)  lmu  ((log(x) - lmu) ./ (lsc .* lsc)) .* ds
@dfunc logpdflogNormal(lmu, lsc::Real, x)   lsc  ( tmp2=lsc.*lsc ; sum( (lmu.*lmu - tmp2 - log(x).*(2lmu-log(x))) ./ (lsc.*tmp2) ) .* ds )
@dfunc logpdflogNormal(lmu, lsc::Array, x)  lsc  ( tmp2=lsc.*lsc ; ( (lmu.*lmu - tmp2 - log(x).*(2lmu-log(x))) ./ (lsc.*tmp2) ) .* ds )

# TODO : find a way to implement multi variate distribs that goes along well with vectorization (Dirichlet, Categorical)
# TODO : other continuous distribs ? : Pareto, Rayleigh, Logistic, Levy, Laplace, Dirichlet, FDist
# TODO : other discrete distribs ? : NegativeBinomial, DiscreteUniform, HyperGeometric, Geometric, Categorical

## Bernoulli distribution (Note : no derivation on x parameter as it is an integer)
@dfunc logpdfBernoulli(p::Real, x)     p     sum(1. ./ (p - (1. - x))) * ds
@dfunc logpdfBernoulli(p::Array, x)    p     (1. ./ (p - (1. - x))) * ds

## Binomial distribution (Note : no derivation on x and n parameters as they are integers)
@dfunc logpdfBinomial(n, p::Real, x)   p    sum(x ./ p - (n-x) ./ (1 - p)) * ds
@dfunc logpdfBinomial(n, p::Array, x)  p    (x ./ p - (n-x) ./ (1 - p)) * ds

## Poisson distribution (Note : no derivation on x parameter as it is an integer)
@dfunc logpdfPoisson(lambda::Real, x)   lambda   sum(x ./ lambda - 1) * ds
@dfunc logpdfPoisson(lambda::Array, x)  lambda   (x ./ lambda - 1) * ds


#  fake distribution to test gradient code
@dfunc logpdfTestDiff(x)    x      +ds


## returns sample value for the given Symobl or Expr (for refs)
hint(v::Symbol) = vhint[v]
hint(v) = v  # should be a value if not a Symbol or an Expression
function hint(v::Expr)
	assert(v.head == :ref, "[hint] unexpected variable $v")
	v.args[1] = :( vhint[$(Expr(:quote, v.args[1]))] )
	eval(v)
end


## Returns gradient expression of opex
function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))  # opex=:(z^x);index=2;dsym=:y
	vs = opex.args[1+index]
	ds = symbol("$DERIV_PREFIX$dsym")
	args = opex.args[2:end]
	
	val = map(hint, args)  # get sample values of args to find correct gradient statement

	fn = symbol("d_$(opex.args[1])_x$index")

	try
		dexp = eval(Expr(:call, fn, val...))

		smap = { symbol("x$i") => args[i] for i in 1:length(args)}
		smap[:ds] = ds
		dexp = substSymbols(dexp, smap)

		# unfold for easier optimization later
	    m = MCMCModel()
	    m.source = :(dummy = $dexp )
		unfold!(m)  
		m.exprs[end] = m.exprs[end].args[2] # remove last assignment

		m.exprs[end] = :( $(symbol("$DERIV_PREFIX$vs")) = $(symbol("$DERIV_PREFIX$vs")) + $(m.exprs[end]) )
		return m.exprs
	catch e 
		error("[derive] Doesn't know how to derive $opex by argument $vs")
	end

end