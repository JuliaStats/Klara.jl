######### logistic regression on binary response  ###########

# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
model = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

m = MCMCLikModel(model, vars=zeros(nbeta))
mg = MCMCLikModelG(model, vars=zeros(nbeta))

b1 = benchmark(()->m.eval(m.init),                 
	           "loglik eval", "binomial 10x1000", 1000)
b2 = benchmark(()->mg.evalg(mg.init),              
	           "loglik and gradient eval", "binomial 10x1000", 1000)
b3 = benchmark(()->run(m * RWM(0.1), steps=1000),  
	           "1000 RWM steps", "binomial 10x1000", 1)

res = [b1;b2;b3]
