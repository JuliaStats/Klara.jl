#### clean session.....

using MCMC

# Model definition, 
#  method 1 = state explictly your functions

mymodel = Model(v-> -dot(v,v), 3, ones(3))  # loglik of Normal distrib, vector of 3, initial values 1.0

# or for a model providing the gradient : 
mymodel2 = ModelG(v-> -dot(v,v), v->(-dot(v,v), -2v), 3, ones(3))  # 2nd function returns a tuple (loglik, gradient)


##### method 2 = using expression parsing and autodiff

modexpr = quote
	v ~ Normal(0, 1)
end

mymodel = Model(modexpr, v=ones(3))  # without gradient
mymodel2 = ModelG(modexpr, v=ones(3))  # with gradient


##### running a single chain

res = mymodel * RWM(0.1) * (100:1000)  # burnin = 99
res.samples  # prints samples
mapslices(mean, res.samples[:beta], 2)
mapslices(std, res.samples[:beta], 2)

res = res * (1:10000)  # continue sampling where it stopped


mymodel * MALA(0.1) * (1:1000) # throws an error because mymodel 
                               #  does not provide the gradient function MALA sampling needs

mymodel2 * MALA(0.1) * (1:1000) # now this works


##### running multiple chains

res = mymodel2 * [RWM(0.1), MALA(0.1), HMC(3,0.1)] * (1:1000) # test all 3 samplers
res[2].samples  # prints samples for MALA(0.1)

res = mymodel2 * [HMC(i,0.1) for i in 1:5] * (1:1000) # test HMC with varying # of inner steps


#### end of example


#######  start of example  2 (seqMC) ##########

using MCMC
using Vega

# We need to define a set of models that converge toward the 
#  distribution of interest (in the spirit of simulated annealing)
nmod = 10  # number of models
p = logspace(1, -1, nmod) 
mods = Model[]
i = 1
while i<=nmod
	m = quote
		y = abs(x)
		y ~ Normal(1, $(p[i]) )
	end

	push!(mods, Model(m, x=0.)) # create MCMCModel
	i += 1
end

mods = Model[]
for fac in logspace(1, -1, 10)
	m = quote
		y = abs(x)
		y ~ Normal(1, $fac )
	end
	println(m)
end

# Plot models
xx = [-3:0.01:3] * ones(nmod)' #'
yy = hcat(map(x->mods[j].eval(x), )
Float64[ mods[j].eval([xx[i,j]]) for i in 1:100, j in 1:nmod]
g = ones(100) * [1:10]'  #'
plot(x = vec(xx), y = exp(vec(yy)), group= vec(g), kind = :line)

# Build MCMCTasks with diminishing scaling
targets = MCMCTask[ mods[i] * RWM(sqrt(p[i])) for i in 1:nmod ]

# Create a 100 particles
particles = [ [randn()] for i in 1:100]

# Launch sequential MC 
# (30 steps x 100 particles = 3000 samples returned in a single MCMCChain)
res = seqMC(targets, particles, steps=30)  

# Plot raw samples
ts = collect(1:10:size(res.samples[:beta],2))
plot(x = ts, y = vec(res.samples[:beta])[ts], kind = :scatter)
plot(x = ts, y = vec(res.samples[:beta])[ts], kind = :line)
# we don't have the real distribution yet because we didn't use the 
#   sample weightings sequential MC produces

# Now resample with replacement using weights
ns = length(res.weights)
cp = cumsum(res.weights) / sum(res.weights)
rs = fill(0, ns)
for n in 1:ns  #  n = 1
	l = rand()
	rs[n] = findfirst(p-> (p>=l), cp)
end
newsamp = vec(res.samples[:beta])[rs]

mean(newsamp)
plot(x = collect(1:ns), y = newsamp, kind = :scatter)


#######  end of example  2  ##########




push!(LOAD_PATH, "P:\\Documents\\julia")

include("P:/Documents/julia/MCMC.jl/src/MCMC.jl")
using MCMC

RWM(0.1)
MALA(1,1000)
whos()

m = Model((v)-> -dot(v,v), 3, ones(3))
m2 = ModelG( v-> -dot(v,v), v->(-dot(v,v), -2v), 3, ones(3) )

m2.eval(m2.init)
m2.evalg(m2.init)

MCMC.spinTask(m, RWM(0.1))

res = m2 * RWM(1.) * (1:1000)
res = m2 * HMC(3, .1) * (1:1000)

res.samples


res = run(m, RWM(1.), steps=10000, burnin=0)
res = run(m2, MALA(1.5), steps=10000, burnin=0)
res = run(m * RWM(0.1), steps=10000, burnin=0)
res = run(m2 * RWM(0.1), steps=10000, burnin=0)
res = run(m2 * MCMC.MALA(0.1), steps=10000, burnin=0)
res = run(m * MCMC.MALA(0.1), steps=10000, burnin=0)


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
model

binom = ModelG(model, vars=ones(10))
test.eval(test.init / 2.)
test.evalg(test.init / 2.)

res = MCMC.run(binom * [MALA(0.1), RWM(0.1), MCMC.HMC], steps=10000, burnin=0)



dump(test)
res[2].samples[:beta]

res[1]


generateModelFunction(model, debug=true, vars=zeros(10))
generateModelFunction(model, debug=true, gradient=true, vars=zeros(10))


test = generateModelFunction(model, gradient=true, vars=zeros(10))


test[1](ones(10))

########################################################


########################################################


# d = {:a => "test", :b => 42}
# collect(keys(d))
# keys

function mymodel(vars::Vector{Float64}=[1.,1])
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

# dump(methods(mymodel, (Vector{Float64},))[1][3])
# v = methods(mymodel, (Vector{Float64},))[1][3]
# isa(v,LambdaStaticData)
# v.sparams
# dump(v)
# mymodel.env.defs

# v2 = Base.uncompressed_ast(v)
# dump(v2)

# :($(Expr(:lambda, 
#          {:vars}, 
#          { {:prob}, 
#            { {:vars, :Any, 0}, 
#              {:prob, :Any, 2}}, 
#            {} 
#          }, 
#          quote  # none, line 2:
#             ~(vars,Normal(0,1.0)) # line 3:
#             prob = /(1,+(1.0,exp(*(X,vars)))) # line 4:
#             return ~(Y,Bernoulli(prob))
#          end
#         ) 
#    )
# )


# test = {:a => PDims(3, (2,2)), :b => PDims(7, (10,)), c: => PDims(1, (2)), :d => PDims(17, tuple())}
# test = {:a => PDims(3, (2,2)), :b => PDims(7, (10,)), :c => PDims(1, (2,)), :d => PDims(17, tuple())}



############  tests seqMC

include("P:/Documents/julia/MCMC.jl/src/MCMC.jl")
using MCMC
using Vega

# define model
p = logspace(-1,0.3,10)
i = 1
mods = Model[]
while i<11
	m = :(v = - (x*x + cos(10*x)) ; v * $(p[i]) )
	println(m)
	push!(mods, Model(m, x=0.))
	i += 1
end
targets = mods * RWM(0.1)

# plot models
xx = linspace(-3,3,100) * ones(10)' #'
yy = [ mods[j].eval([xx[i,j]]) for i in 1:100, j in 1:10]
g = ones(100) * linspace(1,10,10)'  #'
plot(x = vec(xx), y = vec(yy), group= vec(g), kind = :line)


particles = [ [randn()] for i in 1:100]


res = MCMC.seqMC(targets, particles, steps=10, burnin=0)
res = MCMC.seqMC(targets, particles, steps=100, burnin=0)


plot(x = vec(res.samples[:beta]), y = rand(size(res.samples[:beta],2)), kind = :scatter)

#######################################################
include("P:/Documents/julia/MCMC.jl/src/MCMC.jl")
using MCMC
using Vega
# define other model
nmod = 10
p = logspace(1, -1, nmod)
mods = Model[]
i = 1
while i<=nmod
	m = quote
		y = abs(x)
		y ~ Normal(1, $(p[i]) )
	end
	println(m)
	push!(mods, Model(m, x=0.))
	i += 1
end

# plot models
xx = linspace(-3,3,100) * ones(nmod)' #'
yy = Float64[ mods[j].eval([xx[i,j]]) for i in 1:100, j in 1:nmod]
g = ones(100) * linspace(1,10,nmod)'  #'
plot(x = vec(xx), y = exp(vec(yy)), group= vec(g), kind = :line)

targets = MCMCTask[ mods[i] * RWM(sqrt(p[i])) for i in 1:nmod ]
particles = [ [randn()] for i in 1:100]

res = seqMC(targets, particles, steps=30, resTrigger=1e-5)
res = seqMC(targets, particles, steps=30, burnin=0)
res = seqMC(targets, particles, steps=200, burnin=100)

ts = collect(1:1:size(res.samples[:beta],2))
plot(x = ts, y = vec(res.samples[:beta])[ts], kind = :scatter)
plot(x = ts, y = vec(res.samples[:beta])[ts], kind = :line)

# resample using weights
ns = length(res.weights)
cp = cumsum(res.weights) / sum(res.weights)
rs = fill(0, ns)
for n in 1:ns  #  n = 1
	l = rand()
	rs[n] = findfirst(p-> (p>=l), cp)
end
newsamp = vec(res.samples[:beta])[rs]

mean(newsamp)

dump(res)
res.runTime
plot(x = collect(1:ns), y = newsamp, kind = :scatter)

logW = zeros(npart)
oldll = oldll[rs]   #zeros(npart)



mean(res.samples[:beta])


dump(res)




test(x) = x[2] += 1

a = ones(5)
test(a)

a
