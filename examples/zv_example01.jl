using MCMC

# loglik of Normal distribution with gradient, vector of 3, initial values 1.0
mymodel = model(v-> -dot(v,v), v->-2v, init=ones(3));

mychain = run(mymodel * MALA(0.1), steps=10000, burnin=1000);

# Compute ZV-MALA mean estimators based on linear polynomial
linearZvMcmc, linearCoef = linearZv(mychain.samples, mychain.gradients);

# Compute ZV-MALA mean estimators based on quadratic polynomial
quadraticZvMcmc, quadraticCoef = quadraticZv(mychain.samples, mychain.gradients);
