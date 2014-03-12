#########################################################################
#    testing script for simple examples 
#########################################################################

using Distributions, DataFrames

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
res = run(m * RWM(0.05) * SerialMC(100:1000))
res = run(m * HMC(2, 0.1) * SerialMC(100:1000))
res = run(m * NUTS() * SerialMC(100:1000))
res = run(m * MALA(0.001) * SerialMC(100:1000))
# TODO : add other samplers here

# different syntax
res = run(m, RWM(), SerialMC(steps=1000, thinning=10, burnin=0))
res = run(m, HMC(2,0.1), SerialMC(thinning=10, burnin=0))
res = run(m, HMC(2,0.1), SerialMC(burnin=20))



### README examples 

mymodel1 = model(v-> -dot(v,v), init=ones(3))
mymodel2 = model(v-> -dot(v,v), grad=v->-2v, init=ones(3))   

modelxpr = quote
    v ~ Normal(0, 1)
end

mymodel3 = model(modelxpr, v=ones(3))
mymodel4 = model(modelxpr, gradient=true, v=ones(3))

mychain = run(mymodel1, RWM(0.1), SerialMC(steps=1000, burnin=100))
mychain = run(mymodel1, RWM(0.1), SerialMC(steps=1000, burnin=100, thinning=5))
mychain = run(mymodel1, RWM(0.1), SerialMC(101:5:1000))
mychain1 = run(mymodel1 * RWM(0.1) * SerialMC(101:5:1000))

mychain2 = run(mymodel2, HMC(0.75), SerialMC(steps=10000, burnin=1000))

head(mychain2.samples)
head(mychain2.gradients)

acceptance(mychain2)

describe(mychain2)

ess(mychain2)

actime(mychain2)

var(mychain2)
var(mychain2, vtype=:iid)
var(mychain2, vtype=:ipse)
var(mychain2, vtype=:bm)

mychain1 = resume(mychain1, steps=10000)

@test_throws run(mymodel3 * MALA(0.1) * SerialMC(1:1000))

run(mymodel4 * MALA(0.1) * SerialMC(1:1000))

mychain = run(mymodel2 * [RWM(0.1), MALA(0.1), HMC(3,0.1)] * SerialMC(steps=1000)) 
mychain[2].samples

mychain = run(mymodel2 * [HMC(i,0.1) for i in 1:5] * SerialMC(steps=1000))

nmod = 10
mods = Array(MCMCLikModel, nmod)
sts = logspace(1, -1, nmod)
for i in 1:nmod
  m = quote
    y = abs(x)
    y ~ Normal(1, $(sts[i]))
  end
  mods[i] = model(m, x=0)
end

targets = MCMCTask[mods[i] * RWM(sts[i]) * SeqMC(steps=10, burnin=0) for i in 1:nmod]
particles = [[randn()] for i in 1:1000]

mychain3 = run(targets, particles=particles)

mychain4 = wsample(mychain3.samples["x"], mychain3.diagnostics["weigths"], 1000)
mean(mychain4)
