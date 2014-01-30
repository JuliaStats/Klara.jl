#########################################################################
#    Testing script for gradients of distributions
#########################################################################

include("helper_diff.jl")

## variables of different dimension for testing
v0ref = 2.
v1ref = [2., 3, 0.1, 0, -5]
v2ref = [-1. 3 0 ; 0 5 -2]


## continuous distributions
@test_combin    logpdf(Normal(mu, sigma), x)      sigma->sigma<=0?0.1:sigma size(mu)==size(sigma) &&
													(ndims(mu)==0 || size(x)==size(mu))

@test_combin    logpdf(Uniform(a, b), x)    a->a-10 b->b+10 size(a)==size(b) && (ndims(a)==0 || size(x)==size(a))

@test_combin    logpdf(Weibull(sh, sc), x)    sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc  x->x<=0?0.1:x size(sh)==size(sc) && 
													(ndims(sh)==0 || size(x)==size(sh))

@test_combin    logpdf(Gamma(sh, sc), x)    sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc x->x<=0?0.1:x  size(sh)==size(sc) && 
												(ndims(sc)==0 || size(sc)==size(x))

@test_combin    logpdf(Beta(a,b),x)         x->clamp(x, 0.01, 0.99) a->a<=0?0.1:a b->b<=0?0.1:b (size(a)==size(b)) && 
												(ndims(a)==0 || size(x)==size(a))

@test_combin    logpdf(TDist(df),x)         df->df<=0?0.1:df    (size(df)==size(x)) || ndims(df)==0  # fail

@test_combin    logpdf(Exponential(sc),x)   sc->sc<=0?0.1:sc  x->x<=0?0.1:x   (size(sc)==size(x)) || ndims(sc)==0  # fail

@test_combin    logpdf(Cauchy(mu,sc),x)   sc->sc<=0?0.1:sc  (size(mu)==size(sc)) && 
								(ndims(mu)==0 || size(x)==size(mu)) 

@test_combin    logpdf(LogNormal(lmu,lsc),x)   lsc->lsc<=0?0.1:lsc x->x<=0?0.1:x (size(lmu)==size(lsc)) && 
								(ndims(lmu)==0 || size(x)==size(lmu))

# @test_combin    logpdf(Laplace(loc,sc),x)   sc->sc<=0?0.1:sc (size(loc)==size(sc)) && 
# 								(ndims(loc)==0 || size(x)==size(loc))


## discrete distributions
#  the variable x being an integer should not be derived against

# note for Bernoulli : having prob=1 or 0 is ok but will make the numeric differentiator fail => not tested
@test_combin logpdf(Bernoulli(prob),x) prob prob->clamp(prob, 0.01, 0.99) x->float64(x>0) size(prob)==size(x)||ndims(prob)==0  #fail

@test_combin logpdf(Poisson(l),x) l l->l<=0?0.1:l x->iround(abs(x)) size(l)==size(x)||ndims(l)==0  #fail

@test_combin logpdf(Binomial(n,prob),x) prob prob->clamp(prob, 0.01, 0.99) x->iround(abs(x)) n->iround(abs(n)+10) (size(n)==size(prob)) && 
								(ndims(n)==0 || size(x)==size(n)) 


#########################################################################
#   misc. tests
#########################################################################

@test_throws deriv1(:(logpdfBernoulli(1, x)), [0.])
@test_throws deriv1(:(logpdfPoisson(1, x)), [1.])
@test_throws deriv1(:(logpdfBinomial(3, 0.5, x)), [0.])
@test_throws deriv1(:(logpdfBinomial(x, 0.5, 2)), [0.])

##  ref  testing
deriv1(:(x[2]),              v1ref)
deriv1(:(x[2:3]),            v1ref)
deriv1(:(x[2:end]),          v1ref)

deriv1(:(x[2:end]),          v2ref)
deriv1(:(x[2]),              v2ref)
deriv1(:(x[2:4]),            v2ref)
deriv1(:(x[:,2]),            v2ref)
deriv1(:(x[1,:]),            v2ref)
deriv1(:(x[2:end,:]),        v2ref)
deriv1(:(x[:,2:end]),        v2ref)

deriv1(:(x[2]+x[1]),          v2ref)
deriv1(:(log(x[2]^2+x[1]^2)), v2ref)

# fail case when individual elements of an array are set several times
# model = :(x::real(3); y=x; y[2] = x[1] ; y ~ TestDiff())

