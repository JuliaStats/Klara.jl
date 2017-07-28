using Base.Test
using Distributions
using Klara

fields = Dict{Symbol, Symbol}(
  :pdf=>:pdf,
  :prior=>:prior,
  :spdf=>:setpdf,
  :sprior=>:setprior,
  :ll=>:loglikelihood!,
  :lp=>:logprior!,
  :lt=>:logtarget!,
  :gll=>:gradloglikelihood!,
  :glp=>:gradlogprior!,
  :glt=>:gradlogtarget!,
  :tll=>:tensorloglikelihood!,
  :tlp=>:tensorlogprior!,
  :tlt=>:tensorlogtarget!,
  :dtll=>:dtensorloglikelihood!,
  :dtlp=>:dtensorlogprior!,
  :dtlt=>:dtensorlogtarget!,
  :uptoglt=>:uptogradlogtarget!,
  :uptotlt=>:uptotensorlogtarget!,
  :uptodtlt=>:uptodtensorlogtarget!
)

println("    Testing BasicContMuvParameter constructors:")

println("      Initialization via index and key fields...")

p = BasicContMuvParameter(:p, 1, signature=:low)

for field in values(fields)
  @test getfield(p, field) == nothing
end

println("      Initialization via pdf field...")

pv = [5.18, -7.76]
μv = [6.11, -8.5]
states = VariableState[BasicContMuvParameterState(pv), BasicMuvVariableState(μv)]

p = BasicContMuvParameter(:p, 1, signature=:low, pdf=MvNormal(states[2].value, 1.), states=states)

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states[1])
@test states[1].logtarget == lt
p.gradlogtarget!(states[1])
@test states[1].gradlogtarget == glt

states[1] = BasicContMuvParameterState(pv)

p.uptogradlogtarget!(states[1])
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = [-11.87, -13.44]
states[1].value = pv
μv = [-20.2, -18.91]
states[2].value = μv

p.pdf = MvNormal(states[2].value, 1.)

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states[1])
@test states[1].logtarget == lt
p.gradlogtarget!(states[1])
@test states[1].gradlogtarget == glt

states[1] = BasicContMuvParameterState(pv)

p.uptogradlogtarget!(states[1])
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :spdf, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via prior field...")

pv = [1.25, 1.8]
pvlen = length(pv)
σv = [10., 2.]
states = VariableState[BasicContMuvParameterState(pv), BasicMuvVariableState(σv)]

p = BasicContMuvParameter(:p, 1, signature=:low, prior=MvNormal(zeros(pvlen), states[2].value), states=states)

distribution = MvNormal(zeros(pvlen), σv)
p.prior == distribution
p.logprior!(states[1])
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states[1])
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

for field in [
  :pdf, :spdf,
  :sprior,
  :ll, :lt,
  :gll, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end

pv = [-0.21, 0.98]
pvlen = length(pv)
states[1].value = pv
σv = ones(pvlen)
states[2].value = σv

p.prior = MvNormal(zeros(pvlen), states[2].value)

distribution = MvNormal(zeros(pvlen), σv)
p.prior == distribution
p.logprior!(states[1])
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states[1])
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

for field in [
  :pdf, :spdf,
  :sprior,
  :ll, :lt,
  :gll, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via setpdf field...")

pv = [3.79, 4.64]
μv = [5.4, 5.3]
states = VariableState[BasicContMuvParameterState(pv), BasicMuvVariableState(μv)]

p = BasicContMuvParameter(:p, 1, signature=:low, setpdf=(state, states) -> MvNormal(states[2].value, 1.), states=states)
setpdf!(p, states[1])

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states[1])
@test states[1].logtarget == lt
p.gradlogtarget!(states[1])
@test states[1].gradlogtarget == glt

states[1] = BasicContMuvParameterState(pv)

p.uptogradlogtarget!(states[1])
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = [-1.91, -0.9]
states[1].value = pv
μv = [0.12, 0.99]
states[2].value = μv

setpdf!(p, states[1])

distribution = MvNormal(μv, 1.)
p.pdf == distribution
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states[1])
@test states[1].logtarget == lt
p.gradlogtarget!(states[1])
@test states[1].gradlogtarget == glt

states[1] = BasicContMuvParameterState(pv)

p.uptogradlogtarget!(states[1])
@test (states[1].logtarget, states[1].gradlogtarget) == (lt, glt)

for field in [:prior, :sprior, :ll, :lp, :gll, :glp, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

println("      Initialization via setprior field...")

pv = [3.55, 9.5]
pvlen = length(pv)
σv = [2., 10.]
states = VariableState[BasicContMuvParameterState(pv), BasicMuvVariableState(σv)]

p = BasicContMuvParameter(
  :p, 1, signature=:low, setprior=(state, states) -> MvNormal(zeros(pvlen), states[2].value), states=states
)
setprior!(p, states[1])

distribution = MvNormal(zeros(pvlen), σv)
p.prior == distribution
p.logprior!(states[1])
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states[1])
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

pv = [-2.67, 7.71]
states[1].value = pv
σv = [5., 4.]
states[2].value = σv

setprior!(p, states[1])

distribution = MvNormal(zeros(pvlen), σv)
p.prior == distribution
p.logprior!(states[1])
@test states[1].logprior == logpdf(distribution, pv)
p.gradlogprior!(states[1])
@test states[1].gradlogprior == gradlogpdf(distribution, pv)

for field in [:pdf, :spdf, :ll, :lt, :gll, :glt, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptoglt, :uptotlt, :uptodtlt]
  @test getfield(p, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows MvNormal(μ0, σ0)
println("      Initialization via loglikelihood! and logprior! fields...")

μv = [-2.637, -1.132]
μvlen = length(μv)
xv = [-1.88, 2.23]
Σv = eye(μvlen)
μ0v = zeros(μvlen)
Σ0v = eye(μvlen)
states = VariableState[
  BasicContMuvParameterState(μv),
  BasicMuvVariableState(xv),
  BasicMavVariableState(Σv),
  BasicMuvVariableState(μ0v),
  BasicMavVariableState(Σ0v)
]

llf(state, states) =
  state.loglikelihood =
  -0.5*(
    (states[2].value-state.value)'*inv(states[3].value)*(states[2].value-state.value)+
    μvlen*log(2*pi)+
    logdet(states[3].value)
  )[1]

lpf(state, states) =
  state.logprior =
  -0.5*(
    (state.value-states[4].value)'*inv(states[5].value)*(state.value-states[4].value)+
    μvlen*log(2*pi)+
    logdet(states[5].value)
  )[1]

μ = BasicContMuvParameter(:μ, 1, signature=:low, loglikelihood=llf, logprior=lpf, states=states)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
μ.loglikelihood!(states[1])
@test isapprox(states[1].loglikelihood, ll)
μ.logprior!(states[1])
@test isapprox(states[1].logprior, lp)

states[1] = BasicContMuvParameterState(μv)

μ.logtarget!(states[1])
@test isapprox(states[1].loglikelihood, ll)
@test isapprox(states[1].logprior, lp)
@test isapprox(states[1].logtarget, lt)

for field in [
  :pdf, :prior,
  :spdf, :sprior,
  :gll, :glp, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(μ, fields[field]) == nothing
end

# Unnormalized normal target
println("      Initialization via logtarget! field...")

pv = [-1.28, 1.73]
pvlen = length(pv)
μv = [9.4, 3.32]
states = VariableState[BasicContMuvParameterState(pv), BasicMuvVariableState(μv)]

p = BasicContMuvParameter(
  :p,
  1,
  signature=:low,
  logtarget=(state, states) -> state.logtarget = -(state.value-states[2].value)⋅(states[1].value-states[2].value),
  states=states
)

p.logtarget!(states[1])
@test isapprox(0.5*(states[1].logtarget-pvlen*log(2*pi))[1], logpdf(MvNormal(μv, 1.), pv))

for field in [
  :pdf, :prior,
  :spdf, :sprior,
  :ll, :lp,
  :gll, :glp, :glt,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptoglt, :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows MvNormal(μ0, σ0)
println("      Initialization via loglikelihood!, gradloglikelihood! and prior fields...")

μv = [5.59, -7.25]
μvlen = length(μv)
xv = [4.11, 8.17]
Σv = eye(μvlen)
μ0v = zeros(μvlen)
Σ0v = eye(μvlen)
states = VariableState[
  BasicContMuvParameterState(μv),
  BasicMuvVariableState(xv),
  BasicMavVariableState(Σv),
  BasicMuvVariableState(μ0v),
  BasicMavVariableState(Σ0v)
]

llf(state, states) =
  state.loglikelihood =
  -0.5*(
    (states[2].value-states[1].value)'*inv(states[3].value)*(states[2].value-states[1].value)+
    μvlen*log(2*pi)+
    logdet(states[3].value)
  )[1]

gllf(state, states) = states[1].gradloglikelihood = (states[3].value\(states[2].value-states[1].value))

μ = BasicContMuvParameter(
  :μ,
  1,
  signature=:low,
  loglikelihood=llf,
  gradloglikelihood=gllf,
  prior=MvNormal(states[4].value, states[5].value),
  states=states
)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(states[1])
@test isapprox(states[1].loglikelihood, ll)
μ.logprior!(states[1])
@test states[1].logprior == lp
μ.gradloglikelihood!(states[1])
@test isapprox(states[1].gradloglikelihood, gll)
μ.gradlogprior!(states[1])
@test states[1].gradlogprior == glp

states[1] = BasicContMuvParameterState(μv)

μ.logtarget!(states[1])
@test isapprox(states[1].loglikelihood, ll)
@test states[1].logprior == lp
@test isapprox(states[1].logtarget, lt)
μ.gradlogtarget!(states[1])
@test isapprox(states[1].gradloglikelihood, gll)
@test states[1].gradlogprior == glp
@test isapprox(states[1].gradlogtarget, glt)

states[1] = BasicContMuvParameterState(μv)

μ.uptogradlogtarget!(states[1])
@test isapprox(states[1].loglikelihood, ll)
@test states[1].logprior == lp
@test isapprox(states[1].logtarget, lt)
@test isapprox(states[1].gradloglikelihood, gll)
@test states[1].gradlogprior == glp
@test isapprox(states[1].gradlogtarget, glt)

for field in [:pdf, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

μv = [4.21, 7.91]
states[1].value = μv
xv = [-3.1, -2.52]
states[2].value = xv
Σv = diagm([2., 1.])
states[3].value = Σv
μ0v = [1., 2.5]
states[4].value = μ0v
Σ0v = diagm([3., 5.])
states[5].value = Σ0v

μ.prior = MvNormal(states[4].value, states[5].value)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(states[1])
@test isapprox(states[1].loglikelihood, ll)
μ.logprior!(states[1])
@test states[1].logprior == lp
μ.gradloglikelihood!(states[1])
@test isapprox(states[1].gradloglikelihood, gll)
μ.gradlogprior!(states[1])
@test states[1].gradlogprior == glp

states[1] = BasicContMuvParameterState(μv)

μ.logtarget!(states[1])
@test isapprox(states[1].loglikelihood, ll)
@test states[1].logprior == lp
@test isapprox(states[1].logtarget, lt)
μ.gradlogtarget!(states[1])
@test isapprox(states[1].gradloglikelihood, gll)
@test states[1].gradlogprior == glp
@test isapprox(states[1].gradlogtarget, glt)

states[1] = BasicContMuvParameterState(μv)

μ.uptogradlogtarget!(states[1])
@test isapprox(states[1].loglikelihood, ll)
@test states[1].logprior == lp
@test isapprox(states[1].logtarget, lt)
@test isapprox(states[1].gradloglikelihood, gll)
@test states[1].gradlogprior == glp
@test isapprox(states[1].gradlogtarget, glt)

for field in [:pdf, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

# Normal-normal: log-likelihood follows N(μ, σ) and log-prior follows MvNormal(μ0, σ0)
println("      Initialization via loglikelihood!, logprior!, gradloglikelihood! and gradlogprior! fields...")

μv = [6.69, -3.125]
μvlen = length(μv)
xv = [5.43, 9.783]
Σv = eye(μvlen)
μ0v = zeros(μvlen)
Σ0v = eye(μvlen)
states = VariableState[
  BasicContMuvParameterState(μv),
  BasicMuvVariableState(xv),
  BasicMavVariableState(Σv),
  BasicMuvVariableState(μ0v),
  BasicMavVariableState(Σ0v)
]

llf(state, states) =
  state.loglikelihood =
  -0.5*(
    (states[2].value-state.value)'*inv(states[3].value)*(states[2].value-state.value)+
    μvlen*log(2*pi)+
    logdet(states[3].value)
  )[1]

lpf(state, states) =
  state.logprior =
  -0.5*(
    (state.value-states[4].value)'*inv(states[5].value)*(state.value-states[4].value)+
    μvlen*log(2*pi)+
    logdet(states[5].value)
  )[1]

gllf(state, states) = state.gradloglikelihood = (states[3].value\(states[2].value-state.value))

glpf(state, states) = state.gradlogprior = -(states[5].value\(state.value-states[4].value))

μ = BasicContMuvParameter(
  :μ, 1, signature=:low, loglikelihood=llf, logprior=lpf, gradloglikelihood=gllf, gradlogprior=glpf, states=states
)

ld = MvNormal(μv, Σv)
pd = MvNormal(μ0v, Σ0v)
ll, lp = logpdf(ld, xv), logpdf(pd, μv)
lt = ll+lp
gll, glp = -gradlogpdf(ld, xv), gradlogpdf(pd, μv)
glt = gll+glp
μ.loglikelihood!(states[1])
@test isapprox(states[1].loglikelihood, ll)
μ.logprior!(states[1])
@test isapprox(states[1].logprior, lp)
μ.gradloglikelihood!(states[1])
@test isapprox(states[1].gradloglikelihood, gll)
μ.gradlogprior!(states[1])
@test isapprox(states[1].gradlogprior, glp)

states[1] = BasicContMuvParameterState(μv)

μ.logtarget!(states[1])
@test isapprox(states[1].loglikelihood, ll)
@test isapprox(states[1].logprior, lp)
@test isapprox(states[1].logtarget, lt)
μ.gradlogtarget!(states[1])
@test isapprox(states[1].gradloglikelihood, gll)
@test isapprox(states[1].gradlogprior, glp)
@test isapprox(states[1].gradlogtarget, glt)

states[1] = BasicContMuvParameterState(μv)

μ.uptogradlogtarget!(states[1])
@test isapprox(states[1].loglikelihood, ll)
@test isapprox(states[1].logprior, lp)
@test isapprox(states[1].logtarget, lt)
@test isapprox(states[1].gradloglikelihood, gll)
@test isapprox(states[1].gradlogprior, glp)
@test isapprox(states[1].gradlogtarget, glt)

for field in [:pdf, :prior, :spdf, :sprior, :tll, :tlp, :tlt, :dtll, :dtlp, :dtlt, :uptotlt, :uptodtlt]
  @test getfield(μ, fields[field]) == nothing
end

# Unnormalized normal target
println("      Initialization via logtarget! and gradlogtarget! fields...")

pv = [-4.29, 2.91]
μv = [2.2, 2.02]
states = VariableState[BasicContMuvParameterState(pv), BasicMuvVariableState(μv)]

p = BasicContMuvParameter(
  :p,
  1,
  signature=:low,
  logtarget=(state, states) -> state.logtarget = -(state.value-states[2].value)⋅(states[1].value-states[2].value),
  gradlogtarget=(state, states) -> state.gradlogtarget = -2*(state.value-states[2].value),
  states=states
)

distribution = MvNormal(μv, 1.)
lt, glt = logpdf(distribution, pv), gradlogpdf(distribution, pv)
p.logtarget!(states[1])
@test isapprox(0.5*(states[1].logtarget-pvlen*log(2*pi))[1], lt)
p.gradlogtarget!(states[1])
@test isapprox(0.5*states[1].gradlogtarget, glt)

states[1] = BasicContMuvParameterState(pv)

p.uptogradlogtarget!(states[1])
@test isapprox(0.5*(states[1].logtarget-pvlen*log(2*pi))[1], lt)
@test isapprox(0.5*states[1].gradlogtarget, glt)

for field in [
  :pdf, :prior,
  :spdf, :sprior,
  :ll, :lp,
  :gll, :glp,
  :tll, :tlp, :tlt,
  :dtll, :dtlp, :dtlt,
  :uptotlt, :uptodtlt
]
  @test getfield(p, fields[field]) == nothing
end
