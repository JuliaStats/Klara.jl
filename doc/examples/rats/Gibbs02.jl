using Distributions
using Lora

# using Gadfly

nrats, ndays = 30, 5
nvars = 9

a_μ = BasicContUnvParameter(:a_μ, logtarget=(p::Float64, v::Vector) -> logpdf(Normal(0, 100), p), nkeys=nvars)

a_σ2 = BasicContUnvParameter(:a_σ2, logtarget=(p::Float64, v::Vector) -> logpdf(InverseGamma(0.01, 0.01), p), nkeys=nvars)

b_μ = BasicContUnvParameter(:b_μ, logtarget=(p::Float64, v::Vector) -> logpdf(Normal(0, 100), p), nkeys=nvars)

b_σ2 = BasicContUnvParameter(:b_σ2, logtarget=(p::Float64, v::Vector) -> logpdf(InverseGamma(0.01, 0.01), p), nkeys=nvars)

y_σ2 = BasicContUnvParameter(:y_σ2, logtarget=(p::Float64, v::Vector) -> logpdf(InverseGamma(0.01, 0.01), p), nkeys=nvars)

x = Data(:x)

a = BasicContMuvParameter(
  :a,
  logtarget=(p::Vector{Float64}, v::Vector) -> logpdf(MvNormal(v[1]*ones(nrats), sqrt(v[2])), p),
  nkeys=nvars
)

b = BasicContMuvParameter(
  :b,
  logtarget=(p::Vector{Float64}, v::Vector) -> logpdf(MvNormal(v[3]*ones(nrats), sqrt(v[4])), p),
  nkeys=nvars
)

ylogtarget(p::Vector{Float64}, v::Vector) =
  logpdf(MvNormal(repeat(v[7], inner=[ndays])+repeat(v[8], inner=[ndays]).*repeat(v[6], outer=[nrats]), sqrt(v[5])), p)

y = BasicContMuvParameter(:y, logtarget=ylogtarget, nkeys=nvars)

model = GenericModel([a_μ, a_σ2, b_μ, b_σ2, y_σ2, x, a, b, y], isindexed=false)

mcrange = BasicMCRange(nsteps=70000, burnin=50000)

X, = dataset("rats", "age")
X = vec(X)-mean(X)
Y, = dataset("rats", "weight")
v0 = Dict{Symbol, Any}(
  :a_μ=>150.,
  :a_σ2=>1.,
  :b_μ=>10.,
  :b_σ2=>1.,
  :y_σ2=>1.,
  :x=>X,
  :a=>fill(250., nrats),
  :b=>fill(6., nrats),
  :y=>vec(hcat([Y[i, :] for i in 1:30]...))
)

samplers = [
  [
    MH(1.), MH(x::Float64 -> Truncated(Normal(x, 1.), 0, Inf), symmetric=false),
    MH(1.), MH(x::Float64 -> Truncated(Normal(x, 1.), 0, Inf), symmetric=false),
    MH(x::Float64 -> Truncated(Normal(x, 1.), 0, Inf), symmetric=false)
  ];
  fill(MH(1*ones(nrats)), 2)
]

kk = [:a_μ, :a_σ2, :b_μ, :b_σ2, :y_σ2, :a, :b]

samplers = Dict(zip(kk, samplers))

basicmcjobs =
  Dict([(k, BasicMCJob(
    model,
    samplers[k],
    BasicMCRange(nsteps=1, burnin=0),
    v0,
    pindex=model[k].index,
    jointtarget=true,
    resetpstate=false
  )) for k in kk])

tuners = Dict(zip(kk, fill(RobertsRosenthalMCTuner(0.44), 8)))
  
gibbsjob = GibbsJob(model, basicmcjobs, mcrange, v0, jointtarget=true, tuner=tuners, verbose=true)

@time run(gibbsjob)

output(gibbsjob)

chains = Dict(gibbsjob)

mean(chains[:b_μ].value)

# plot(x=1:length(chains[:b_μ].value), y=chains[:b_μ].value, Geom.line)
