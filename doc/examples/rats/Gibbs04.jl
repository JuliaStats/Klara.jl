using Distributions
using Lora

# using Gadfly

dpkeys = [:am, :assq, :bm, :bssq, :yssq, :a, :b, :y]
ndpkeys = length(dpkeys)
nudpkeys = ndpkeys-1

nrats, ndays = 30, 5
nvars = 9  
  
X, = dataset("rats", "age")
X = vec(X)-mean(X)
Y, = dataset("rats", "weight")
v0 = Any[150., 1., 10., 1., 1., fill(250., nrats), fill(6., nrats), vec(hcat([Y[i, :] for i in 1:30]...)), X]

logtargets = Dict{Symbol, Function}(
  :am => p -> logpdf(Normal(0, 100), p),
  :assq => p -> logpdf(InverseGamma(0.01, 0.01), p),
  :bm => p -> logpdf(Normal(0, 100), p),
  :bssq => p -> logpdf(InverseGamma(0.01, 0.01), p),
  :yssq => p -> logpdf(InverseGamma(0.01, 0.01), p),
  :a => (p, am, assq) -> logpdf(MvNormal(am*ones(nrats), sqrt(assq)), p),
  :b => (p, bm, bssq) -> logpdf(MvNormal(bm*ones(nrats), sqrt(bssq)), p),
  :y => (p, yssq, a, b, x) -> logpdf(
    MvNormal(repeat(a, inner=[ndays])+repeat(b, inner=[ndays]).*repeat(x, outer=[nrats]), sqrt(yssq)),
      p
    )
)

function logtarget(am, assq, bm, bssq, yssq, a, b, y, x)
  logtargets[:am](am)+
  logtargets[:assq](assq)+
  logtargets[:bm](bm)+
  logtargets[:bssq](bssq)+
  logtargets[:yssq](yssq)+
  logtargets[:a](a, am, assq)+
  logtargets[:b](b, bm, bssq)+
  logtargets[:y](y, yssq, a, b, x)
end

nmcmc = 1000
nburnin = 500
npostburnin = nmcmc-nburnin

ns = Dict{Symbol, Int}(
  :am => 1,
  :assq => 1,
  :bm => 1,
  :bssq => 1,
  :yssq => 1,
  :a => 30,
  :b => 30 
)

chains = Dict{Symbol, Array{Float64}}(
  :am => Array(Float64, npostburnin),
  :assq => Array(Float64, npostburnin),
  :bm => Array(Float64, npostburnin),
  :bssq => Array(Float64, npostburnin),
  :yssq => Array(Float64, npostburnin),
  :a => Array(Float64, npostburnin, 2),
  :b => Array(Float64, npostburnin, 2) 
)

oldsample = deepcopy(v0)
delete!(oldsample, :x)

newsample = deepcopy(oldsample)

oldlogtarget = logtarget(Any[v0[k] for k in dpkeys]..., v0[:x])
newlogtarget = NaN

s = Dict{Symbol, Array{Float64}}(
  :am => 1.,
  :assq => 1.,
  :bm => 1.,
  :bssq => 1.,
  :yssq => 1.,
  :a => 1.,
  :b => 1. 
)

type RobertsRosenthalMCTune
  σ::Real # Stepsize of MCMC iteration (for ex leapfrog in HMC or drift stepsize in MALA)
  myflag::Bool
  accepted::Int # Number of accepted MCMC samples during current tuning period
  proposed::Int # Number of proposed MCMC samples during current tuning period
  totproposed::Int # Total number of proposed MCMC samples during burnin
  rate::Real # Observed acceptance rate over current tuning period

  function RobertsRosenthalMCTune(σ::Real, myflag::Bool, accepted::Int, proposed::Int, totproposed::Int, rate::Real)
    @assert σ > 0 "Stepsize of MCMC iteration should be positive"
    @assert accepted >= 0 "Number of accepted MCMC samples should be non-negative"
    @assert proposed >= 0 "Number of proposed MCMC samples should be non-negative"
    @assert totproposed >= 0 "Total number of proposed MCMC samples should be non-negative"
    if !isnan(rate)
      @assert 0 < rate < 1 "Observed acceptance rate should be between 0 and 1"
    end
    new(σ, myflag, accepted, proposed, totproposed, rate)
  end
end

RobertsRosenthalMCTune(σ::Real=1., myflag::Bool=true, accepted::Int=0, proposed::Int=0, totproposed::Int=0) =
  RobertsRosenthalMCTune(σ, myflag, accepted, proposed, totproposed, NaN)

immutable RobertsRosenthalMCTuner
  targetrate::Real # Target acceptance rate
  period::Int # Tuning period over which acceptance rate is computed
  verbose::Bool # If verbose=false then the tuner is silent, else it is verbose

  function RobertsRosenthalMCTuner(targetrate::Real, period::Int, verbose::Bool)
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert period > 0 "Tuning period should be positive"
    new(targetrate, period, verbose)
  end
end

RobertsRosenthalMCTuner(
  targetrate::Real;
  period::Int=100,
  verbose::Bool=false
) =
  RobertsRosenthalMCTuner(targetrate, period, verbose)

function tune!(tune::RobertsRosenthalMCTune, tuner::RobertsRosenthalMCTuner)
  delta = min(0.01, (tune.totproposed/tuner.period)^-0.5)
  epsilon = tune.rate < tuner.targetrate ? -delta : delta
  tune.σ *= exp(epsilon)
end

function reset_burnin!(tune::RobertsRosenthalMCTune)
  tune.totproposed += tune.proposed
  (tune.accepted, tune.proposed, tune.rate) = (0, 0, NaN)
end

rate!(tune::RobertsRosenthalMCTune) = (tune.rate = tune.accepted/tune.proposed)

tune = Dict(zip(dpkeys, fill(RobertsRosenthalMCTune(), ndpkeys)))
delete!(tune, :y)

m = 1

for i in 1:nmcmc
  println(i)
  
  for j in 1:nudpkeys
    k = dpkeys[j]
    newsample[k] = oldsample[k]+tune[k].σ*randn(ns[k])   
    
    ratio = logtarget(Any[v0[k] for k in dpkeys]..., v0[:x])
    
    if accept
      oldsample[k] = copy(newsample[k])
    end
  end
  
  if i < nburnin
  else
    chains[k][m] = oldsample[k]
    m += 1
  end
end
