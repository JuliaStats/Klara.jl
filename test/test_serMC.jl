using Distributions, MCMC
srand(123)

modelexpr = quote
  v ~ Normal(0, 1)
end

mymodel1 = model(v-> -dot(v/2,v/2), init=ones(1))  
mymodel2 = model(v-> -dot(v,v), grad=v->-2v, init=ones(1))   

mymodel = model(modelexpr, gradient=true, v=ones(1))
mychain = run(mymodel1 * RWM() * SerialMC(1:1:20))

describe(mychain)
#mychain.samples

