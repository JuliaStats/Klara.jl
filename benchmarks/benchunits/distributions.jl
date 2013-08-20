######### multiple distributions  ###########
using Distributions

const VECTOR_SIZE = 1000
v = ones(VECTOR_SIZE)

function bench(ex::Expr)  # ex = :(Weibull(1, 1))
	model = Expr(:block, :(y = x * v ; y ~ $ex))

	# to get valid intial values, get the mean of the distribution
	exactMean = eval( :(mean($ex)) )
	# some distribs (Cauchy) have no defined mean
	exactMean = isfinite(exactMean) ? exactMean : 1.0  

	# exactStd = eval( :(std($ex)) )
	# exactStd = isfinite(exactStd) ? exactStd : 1.0

	m = MCMCLikModel(model, x=exactMean)
	mg = MCMCLikModelG(model, x=exactMean)

	name = "$ex on vector of $VECTOR_SIZE"

	b1 = benchmark(()-> m.eval(m.init),     "loglik eval", name, 100)
	b2 = benchmark(() -> mg.evalg(mg.init), "loglik and gradient eval", name, 100)
	b3 = benchmark(() -> run(m * RWM(0.1), steps=100),  
		           "100 RWM steps", name, 10)

	[b1;b2;b3]
end

distribs = [:(Normal(1, 1)),
            :(Normal(3, 12)),
            :(Weibull(1, 1)),
            :(Weibull(3, 1)),
            :(Uniform(0, 2)),
            :(TDist(2.2)),
            :(TDist(4)),
            :(Beta(1,2)),
            :(Beta(3,2)),
            :(Gamma(1,2)),
            :(Gamma(3,0.2)),
            :(Cauchy(0,1)),
            :(Cauchy(-1,0.2)),
            :(Exponential(3)),
            :(Exponential(0.2)),
            :(logNormal(-1, 1)),
            :(logNormal(2, 0.1))]

res = mapreduce(bench, vcat, distribs)

