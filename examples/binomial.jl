######### logistic regression on 1000 obs x 10 var  ###########

using DataFrames
using MCMC

# generate a random dataset
srand(1)
const n = 1000
const nbeta = 10 # number of covariates, including intercept

const X = [ones(n) randn((n, nbeta-1))]  # covariates

const beta0 = randn((nbeta,))
const Y = rand(n) .< ( 1 ./ (1. + exp(- X * beta0))) # logistic response

# define model
ex = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(- X * vars)) 
	Y ~ Bernoulli(prob)
end

m = model(ex, vars=zeros(nbeta), gradient=true)

# run random walk metropolis (10000 steps, 1000 for burnin)
res = run(m * RWM(0.05), steps=1000:10000)

head(res.samples)
[colwise(mean, res.samples) beta0 ] # mean sample vs original coefs

# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 2 inner steps, 0.1 inner step size)
res = run(m * HMC(2, 0.1), steps=1000:10000)

# run NUTS HMC (10000 steps, 1000 for burnin)
res = run(m * NUTS(), steps=1000:10000)
