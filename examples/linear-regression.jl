######### linear regression 1000 obs x 10 var  ###########

using DataFrames
using MCMC

# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of covariates
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = X * beta0 + randn((n,))

# define model
ex = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

m = model(ex, vars=zeros(nbeta))

# run random walk metropolis (10000 steps, 1000 for burnin, thinning 10)
# no adaptation
res = run(m * RWM(0.05), steps=10000:10:100000)

mean(res.diagnostics["accept"])  #  ~ 3%, too low

# with adaptation (target acceptance = 30%)
res = run(m * RWM(tuner=RAM(0.3)), steps=10000:10:100000)

mean(res.diagnostics["accept"])  #  ~ 29.7%, Ok

[colwise(mean, res.samples) beta0 ] # mean sample vs original coefs


