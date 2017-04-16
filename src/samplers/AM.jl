### The implementation of AM in Klara uses the mixture proposal of the following article:
### Gareth O. Roberts and Jeffrey S. Rosenthal
### Examples of Adaptive MCMC
### Journal of Computational and Graphical Statistics, 2012, 18 (2), pp 349-367

### The original adaptive Metropolis algorithm was introduced by the following article:
### Heikki Haario, Eero Saksman and Johanna Tamminen
### An Adaptive Metropolis Algorithm
### Bernoulli, 2001, 7 (2), pp 223-242

### Abstract AM state

abstract AMState{F<:VariateForm} <: MHSamplerState{F}

### AM state subtypes

## UnvAMState holds the internal state ("local variables") of the AM sampler for univariate parameters

type UnvAMState <: AMState{Univariate}
  proposal::UnivariateGMM
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by AM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  lastmean::Real
  secondlastmean::Real
  C0::Real
  C::Real
  w::RealVector
  count::Integer

  function UnvAMState(
    proposal::UnivariateGMM,
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    lastmean::Real,
    secondlastmean::Real,
    C0::Real,
    C::Real,
    w::RealVector,
    count::Integer
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(w[1])
      @assert w[1] > 0 "Weight of core mixture component must be positive"
    end
    if !isnan(w[2])
      @assert w[2] >= 0 "Weight of minor mixture component must be non-negative"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(proposal, pstate, tune, ratio, lastmean, secondlastmean, C0, C, w, count)
  end
end

UnvAMState(
  proposal::UnivariateGMM,
  pstate::ParameterState{Continuous, Univariate},
  C::Real,
  w::RealVector,
  tune::MCTunerState=BasicMCTune(),
  lastmean::Real=NaN
) =
  UnvAMState(proposal, pstate, tune, NaN, lastmean, NaN, C, C, w, 0)

## MuvAMState holds the internal state ("local variables") of the AM sampler for multivariate parameters

type MuvAMState <: AMState{Multivariate}
  proposal::MultivariateGMM
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by AM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  lastmean::RealVector
  secondlastmean::RealVector
  C::RealMatrix
  w::RealVector
  count::Integer

  function MuvAMState(
    proposal::MultivariateGMM,
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    lastmean::RealVector,
    secondlastmean::RealVector,
    C::RealMatrix,
    w::RealVector,
    count::Integer
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(w[1])
      @assert w[1] > 0 "Weight of core mixture component must be positive"
    end
    if !isnan(w[2])
      @assert w[2] >= 0 "Weight of minor mixture component must be non-negative"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(proposal, pstate, tune, ratio, lastmean, secondlastmean, C, w, count)
  end
end

MuvAMState(
  proposal::MultivariateGMM,
  pstate::ParameterState{Continuous, Multivariate},
  C::RealMatrix,
  w::RealVector,
  tune::MCTunerState=BasicMCTune(),
  lastmean::RealVector=Array(eltype(pstate), pstate.size)
) =
  MuvAMState(proposal, pstate, tune, NaN, lastmean, Array(eltype(pstate), pstate.size), C, w, 0)

### Adaptive Metropolis (AM) sampler

immutable AM <: MHSampler
  C0::RealMatrix # Initial covariance of proposal according to best prior knowledge
  corescale::Real # Scaling factor of covariance matrix of the core mixture component
  minorscale::Real # Scaling factor of covariance matrix of the stabilizing mixture component
  c::Real # Non-negative constant with relative small value that determines the mixture weight of the stabilizing component

  function AM(C0::RealMatrix, corescale::Real, minorscale::Real, c::Real)
    @assert 0 < corescale "Constant corescale must be positive, got $corescale"
    @assert 0 < minorscale "Constant minorscale must be positive, got $minorscale"
    @assert 0 <= c "Constant c must be non-negative"
    new(C0, corescale, minorscale, c)
  end
end

AM(C0::RealMatrix; corescale::Real=1., minorscale::Real=1., c::Real=0.05) = AM(C0, corescale, minorscale, c)

AM(C0::RealVector; corescale::Real=1., minorscale::Real=1., c::Real=0.05) =
  AM(diagm(C0), corescale=corescale, minorscale=minorscale, c=c)

AM(C0::Real, d::Integer=1; corescale::Real=1., minorscale::Real=1., c::Real=0.05) =
  AM(fill(C0, d), corescale=corescale, minorscale=minorscale, c=c)

### Initialize AM sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::AM,
  outopts::Dict
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array(Any, length(pstate.diagnostickeys))
  end
end

## Initialize AM state

setproposal(sampler::AM, pstate::ParameterState{Continuous, Univariate}, w::RealVector) =
  UnivariateGMM(
    [pstate.value, pstate.value],
    Float64[sqrt(sampler.corescale*sampler.C0[1, 1]), sqrt(sampler.minorscale)],
    Categorical(w)
  )

setproposal!(sstate::UnvAMState, sampler::AM, pstate::ParameterState{Continuous, Univariate}) =
  sstate.proposal = UnivariateGMM(
    [pstate.value, pstate.value],
    Float64[sqrt(sampler.corescale*sstate.C0), sqrt(sampler.minorscale)],
    Categorical(sstate.w)
  )

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::AM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
)
  w = [1-sampler.c, sampler.c]

  UnvAMState(
    setproposal(sampler, pstate, w),
    generate_empty(pstate),
    sampler.C0[1, 1],
    w,
    tuner_state(parameter, sampler, tuner)
  )
end

setproposal(sampler::AM, pstate::ParameterState{Continuous, Multivariate}, w::RealVector) =
  MixtureModel(
    [MvNormal(pstate.value, sampler.corescale*sampler.C0), MvNormal(pstate.value, sampler.minorscale*eye(pstate.size))], w
  )

setproposal!(sstate::MuvAMState, sampler::AM, pstate::ParameterState{Continuous, Multivariate}) =
  sstate.proposal = setproposal(sampler, pstate, sstate.w)

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::AM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  w = [1-sampler.c, sampler.c]

  MuvAMState(
    setproposal(sampler, pstate, w),
    generate_empty(pstate),
    copy(sampler.C0),
    w,
    tuner_state(parameter, sampler, tuner)
  )
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
  setproposal!(sstate, sampler, pstate)
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
  setproposal!(sstate, sampler, pstate)
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
