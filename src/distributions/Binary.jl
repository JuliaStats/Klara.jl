struct Binary <: DiscreteUnivariateDistribution
  a::Int # Failure
  b::Int # Success
  p::Float64

  function Binary(a::Real, b::Real, p::Real)
    # @check_args(Binary, a < b && zero(p) <= p <= one(p)) # Possibly a bug in Distributions, this line is not working
    new(a, b, p)
  end

  function Binary(a::Real, b::Real)
    # @check_args(Binary, a < b) # Possibly a bug in Distributions, this line is not working
    new(a, b, 0.5)
  end

  Binary() = new(0, 1, 0.5)
end

@distr_support Binary d.a d.b

### Parameters

succprob(d::Binary) = d.p
failprob(d::Binary) = 1.0 - d.p

params(d::Binary) = (d.a, d.b, d.p)

### Properties

mean(d::Binary) = d.a*failprob(d)+d.b*succprob(d)

### Evaluation

pdf(d::Binary, x::Int) = x == d.a ? failprob(d) : x == d.b ? succprob(d) : 0.0

pdf(d::Binary) = Float64[failprob(d), succprob(d)]

logpdf(d::Binary, x::Int) = x == d.a ? log(failprob(d)) : x == d.b ? log(succprob(d)) : -Inf

### Sampling

rand(d::Binary) = rand() <= succprob(d) ? d.a : d.b
