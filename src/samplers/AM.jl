### The implementation of AM in Klara uses the mixture proposal of the following article:
### Gareth O. Roberts and Jeffrey S. Rosenthal
### Examples of Adaptive MCMC
### Journal of Computational and Graphical Statistics, 2012, 18 (2), pp 349-367

### The original adaptive Metropolis algorithm was introduced by the following article:
### Heikki Haario, Eero Saksman and Johanna Tamminen
### An Adaptive Metropolis Algorithm
### Bernoulli, 2001, 7 (2), pp 223-242

### Abstract AM state

abstract type AMState{F<:VariateForm} <: MHSamplerState{F} end

### AM state subtypes

## UnvAMState holds the internal state ("local variables") of the AM sampler for univariate parameters

mutable struct UnvAMState <: AMState{Univariate}
  proposal::Union{UnivariateGMM, RealNormal}
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by AM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  lastmean::Real
  secondlastmean::Real
  C::Real
  sqrtminorscale::Real
  w::RealVector
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function UnvAMState(
    proposal::Union{UnivariateGMM, RealNormal},
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    lastmean::Real,
    secondlastmean::Real,
    C::Real,
    sqrtminorscale::Real,
    w::RealVector,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(sqrtminorscale)
      @assert sqrtminorscale >= 0 "Scaling of stabilizing covariance should be non-negative"
    end
    if !isnan(w[1])
      @assert w[1] > 0 "Weight of core mixture component must be positive"
    end
    if !isnan(w[2])
      @assert w[2] >= 0 "Weight of minor mixture component must be non-negative"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(proposal, pstate, tune, ratio, lastmean, secondlastmean, C, sqrtminorscale, w, count, diagnosticindices)
  end
end

UnvAMState(
  proposal::Union{UnivariateGMM, RealNormal},
  pstate::ParameterState{Continuous, Univariate},
  C::Real,
  sqrtminorscale::Real,
  w::RealVector,
  tune::MCTunerState=BasicMCTune(),
  lastmean::Real=NaN
) =
  UnvAMState(proposal, pstate, tune, NaN, lastmean, lastmean, C, sqrtminorscale, w, 0, Dict{Symbol, Integer}())

## MuvAMState holds the internal state ("local variables") of the AM sampler for multivariate parameters

mutable struct MuvAMState <: AMState{Multivariate}
  proposal::Union{MultivariateGMM, AbstractMvNormal}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by AM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  lastmean::RealVector
  secondlastmean::RealVector
  C::RealMatrix
  sqrtminorscale::Real
  w::RealVector
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function MuvAMState(
    proposal::Union{MultivariateGMM, AbstractMvNormal},
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    lastmean::RealVector,
    secondlastmean::RealVector,
    C::RealMatrix,
    sqrtminorscale::Real,
    w::RealVector,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(sqrtminorscale)
      @assert sqrtminorscale >= 0 "Scaling of stabilizing covariance should be non-negative"
    end
    if !isnan(w[1])
      @assert w[1] > 0 "Weight of core mixture component must be positive"
    end
    if !isnan(w[2])
      @assert w[2] >= 0 "Weight of minor mixture component must be non-negative"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(proposal, pstate, tune, ratio, lastmean, secondlastmean, C, sqrtminorscale, w, count, diagnosticindices)
  end
end

MuvAMState(
  proposal::Union{MultivariateGMM, AbstractMvNormal},
  pstate::ParameterState{Continuous, Multivariate},
  C::RealMatrix,
  sqrtminorscale::Real,
  w::RealVector,
  tune::MCTunerState=BasicMCTune(),
  lastmean::RealVector=Array{eltype(pstate)}(pstate.size)
) =
  MuvAMState(proposal, pstate, tune, NaN, lastmean, lastmean, C, sqrtminorscale, w, 0, Dict{Symbol, Integer}())

### Adaptive Metropolis (AM) sampler

struct AM <: MHSampler
  C0::RealMatrix # Initial covariance of proposal according to best prior knowledge
  corescale::Real # Scaling factor of covariance matrix of the core mixture component
  minorscale::Real # Scaling factor of covariance matrix of the stabilizing mixture component
  c::Real # Non-negative constant with relative small value that determines the mixture weight of the stabilizing component
  t0::Integer

  function AM(C0::RealMatrix, corescale::Real, minorscale::Real, c::Real, t0::Integer)
    @assert 0 < corescale "Constant corescale must be positive, got $corescale"
    @assert 0 < minorscale "Constant minorscale must be positive, got $minorscale"
    @assert 0 <= c "Constant c must be non-negative"
    @assert t0 > 0 "t0 is not positive"
    new(C0, corescale, minorscale, c, t0)
  end
end

AM(C0::RealMatrix; corescale::Real=1., minorscale::Real=1., c::Real=0.05, t0::Integer=10) =
  AM(C0, corescale, minorscale, c, t0)

AM(C0::RealVector; corescale::Real=1., minorscale::Real=1., c::Real=0.05, t0::Integer=10) =
  AM(diagm(C0), corescale=corescale, minorscale=minorscale, c=c, t0=t0)

AM(C0::Real, d::Integer=1; corescale::Real=1., minorscale::Real=1., c::Real=0.05, t0::Integer=10) =
  AM(fill(C0, d), corescale=corescale, minorscale=minorscale, c=c, t0=t0)

### Initialize AM sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::AM,
  outopts::Dict
) where F<:VariateForm
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

## Initialize AM state

set_gmm(sampler::AM, pstate::ParameterState{Continuous, Univariate}, C::Real, sqrtminorscale::Real, w::RealVector) =
  UnivariateGMM([pstate.value, pstate.value], Float64[sqrt(sampler.corescale*C), sqrtminorscale], Categorical(w))

set_gmm!(sstate::UnvAMState, sampler::AM, pstate::ParameterState{Continuous, Univariate}) =
  sstate.proposal = set_gmm(sampler, pstate, sstate.C, sstate.sqrtminorscale, sstate.w)

set_normal(sampler::AM, pstate::ParameterState{Continuous, Univariate}, sqrtminorscale::Real) =
  Normal(pstate.value, sqrtminorscale)

set_normal!(sstate::UnvAMState, sampler::AM, pstate::ParameterState{Continuous, Univariate}) =
  sstate.proposal = set_normal(sampler, pstate, sstate.sqrtminorscale)

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::AM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sqrtminorscale = sqrt(sampler.minorscale)
  w = [1-sampler.c, sampler.c]

  sstate = UnvAMState(
    set_normal(sampler, pstate, sqrtminorscale),
    generate_empty(pstate),
    sampler.C0[1, 1],
    sqrtminorscale,
    w,
    tuner_state(parameter, sampler, tuner),
    pstate.value
  )

  set_diagnosticindices!(sstate, [:accept], diagnostickeys)

  sstate
end

set_gmm(sampler::AM, pstate::ParameterState{Continuous, Multivariate}, C::RealMatrix, sqrtminorscale::Real, w::RealVector) =
  MixtureModel([MvNormal(pstate.value, sampler.corescale*C), MvNormal(pstate.value, sqrtminorscale)], w)

set_gmm!(sstate::MuvAMState, sampler::AM, pstate::ParameterState{Continuous, Multivariate}) =
  sstate.proposal = set_gmm(sampler, pstate, sstate.C, sstate.sqrtminorscale, sstate.w)

set_normal(sampler::AM, pstate::ParameterState{Continuous, Multivariate}, sqrtminorscale::Real) =
  MvNormal(pstate.value, sqrtminorscale)

set_normal!(sstate::MuvAMState, sampler::AM, pstate::ParameterState{Continuous, Multivariate}) =
  sstate.proposal = set_normal(sampler, pstate, sstate.sqrtminorscale)

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::AM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sqrtminorscale = sqrt(sampler.minorscale)
  w = [1-sampler.c, sampler.c]

  sstate = MuvAMState(
    set_normal(sampler, pstate, sqrtminorscale),
    generate_empty(pstate),
    copy(sampler.C0),
    sqrtminorscale,
    w,
    tuner_state(parameter, sampler, tuner),
    copy(pstate.value)
  )

  set_diagnosticindices!(sstate, [:accept], diagnostickeys)

  sstate
end

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::AM
)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::AM
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

## Reset sampler state

function reset!(
  sstate::UnvAMState,
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::AM,
  tuner::MCTuner
)
  set_normal!(sstate, sampler, pstate)
  reset!(sstate.tune, sampler, tuner)
  sstate.lastmean = pstate.value
  sstate.C0 = sampler.C0[1, 1]
  sstate.C = sstate.C0
  sstate.count = 0
end

function reset!(
  sstate::MuvAMState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::AM,
  tuner::MCTuner
)
  set_normal!(sstate, sampler, pstate)
  reset!(sstate.tune, sampler, tuner)
  sstate.lastmean = copy(pstate.value)
  sstate.C = copy(sampler.C0)
  sstate.count = 0
end

show(io::IO, sampler::AM) = print(
  io,
  "AM sampler: scaling of core covariance = ",
  sampler.corescale,
  ", scaling of stabilizing covariance = ",
  sampler.minorscale,
  ", weight of stabilizing mixture component = ",
  sampler.c
)
