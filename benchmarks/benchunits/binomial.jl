######### logistic regression on binary response  ###########

# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
ex = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

m = model(ex, vars=zeros(nbeta))
mg = model(ex, gradient=true, vars=zeros(nbeta))

name = "binomial 10x1000"

f1 = ()-> m.eval(m.init)
f2 = ()-> mg.evalg(mg.init)
f3 = ()-> run(m * RWM(0.1), steps=100)

res = [	benchmark( f1, "loglik eval", name, 1000) ;
		benchmark( f2, "loglik and gradient eval", name, 1000) ;
		benchmark( f3, "100 RWM steps", name, 10) ]

