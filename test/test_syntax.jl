#########################################################################
#    testing script for simple examples 
#########################################################################

using Base.Test
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





### README examples 

mymodel = model(v-> -dot(v,v), init=ones(3))  
mymodel2 = model(v-> -dot(v,v), grad = v->-2v, init=ones(3))   

modexpr = quote
    v ~ Normal(0, 1)
end

mymodel = model(modexpr, v=ones(3)) # without gradient
mymodel2 = model(modexpr, gradient=true, v=ones(3)) # with gradient

res = run(mymodel * RWM(0.1), steps=1000, burnin=100)
res = run(mymodel * RWM(0.1), steps=1000, burnin=100, thinning=5)
res = run(mymodel * RWM(0.1), steps=101:5:1000)
res = mymodel * RWM(0.1) * (101:5:1000)  

head(res.samples)
describe(res.samples)
head(res.diagnostics)

res = run(res, steps=10000)  


@test_throws mymodel * MALA(0.1) * (1:1000) # throws an error 

mymodel2 * MALA(0.1) * (1:1000) # now this works

res = run(mymodel2 * [RWM(0.1), MALA(0.1), HMC(3,0.1)], steps=1000) 
res[2].samples  # prints samples for MALA(0.1)

res = run(mymodel2 * [HMC(i,0.1) for i in 1:5], steps=1000)

nmod = 10  # number of models
mods = Array(MCMCLikModel, nmod)
sts = logspace(1, -1, nmod)
for i in 1:nmod
	m = quote
		y = abs(x)
		y ~ Normal(1, $(sts[i]) )
	end
	mods[i] = model(m, x=0)
end

targets = MCMCTask[ mods[i] * RWM(sts[i]) for i in 1:nmod ]
particles = [ [randn()] for i in 1:1000]

res = seqMC(targets, particles, steps=10, burnin=0)  

