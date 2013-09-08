#########################################################################
#    testing script for simple examples 
#########################################################################

using MCMC

# generate a random dataset
srand(1)
n = 1000
nbeta = 10 
X = [ones(n) randn((n, nbeta-1))] 
beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
ex = quote
	vars ~ Normal(0, 1.0)  
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

m = model(ex, vars=zeros(nbeta), gradient=true)

# different samplers
res = run(m * RWM(0.05), steps=100:1000)
res = run(m * HMC(2, 0.1), steps=100:1000)
res = run(m * NUTS(), steps=100:1000)
res = run(m * MALA(0.001), steps=100:1000)
# TODO : add other samplers here

# different syntax
res = m * RWM() * (100:1000)
res = run(m * RWM(), steps=1000, thinning=10, burnin=0)
res = run(m, HMC(2,0.1), thinning=10, burnin=0)
res = run(m, HMC(2,0.1), burnin=20)

