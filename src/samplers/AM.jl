### Reference:
### Heikki Haario, Eero Saksman and Johanna Tamminen
### An Adaptive Metropolis Algorithm
### Bernoulli, 2001, 7 (2), pp 223-242

### Abstract AM state

abstract AMState{F<:VariateForm} <: MHSamplerState{F}

### AM state subtypes

## MuvAMState holds the internal state ("local variables") of the AM sampler for multivariate parameters

type MuvAMState <: AMState{Multivariate}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by AM
  tune::MCTunerState
  ratio::Real # Acceptance ratio
  lastmean::RealVector
  secondlastmean::RealVector
  C::RealMatrix
  cholC::RealLowerTriangular
  count::Integer

  function MuvAMState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    lastmean::RealVector,
    secondlastmean::RealVector,
    C::RealMatrix,
    cholC::RealLowerTriangular,
    count::Integer
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(pstate, tune, ratio, lastmean, secondlastmean, C, cholC, count)
  end
end

MuvAMState(
  pstate::ParameterState{Continuous, Multivariate},
  tune::MCTunerState=BasicMCTune(),
  lastmean::RealVector=Array(eltype(pstate), pstate.size),
  C::RealMatrix=Array(eltype(pstate), pstate.size, pstate.size),
  cholC::RealLowerTriangular=chol(C, Val{:L})
) =
  MuvAMState(pstate, tune, NaN, lastmean, Array(eltype(pstate), pstate.size), C, cholC, 0)

### Adaptive Metropolis (AM) sampler

immutable AM <: MHSampler
  C0::RealMatrix # Initial covariance of proposal according to best prior knowledge
  t0::Integer # Number of initial Monte Carlo steps for which C0 will be used
  sd::Real # Constant that depends only on the dimension of the parameter space
  ε::Real # Non-negative constant with relatively small value

  function AM(C0::RealMatrix, t0::Integer, sd::Real, ε::Real)
    @assert 0 < t0 "Number of initial Monte Carlo steps for which C0 will be used must be positive"
    @assert 0 < sd "Constant sd must be positive"
    @assert 0 <= ε "Constant ε must be non-negative"
    new(C0, t0, sd, ε)
  end
end

AM(C0::RealMatrix, d::Integer; t0::Real=3, ε::Real=0.05) = AM(C0, t0, abs2(2.4)/d, ε)

AM(C0::RealVector, d::Integer; t0::Real=3, ε::Real=0.05) = AM(diagm(C0), d, t0=t0, ε=ε)

AM(C0::Real=1., d::Integer=1; t0::Real=3, ε::Real=0.05) = AM(fill(C0, d), d, t0=t0, ε=ε)

### Initialize AM sampler

## Initialize parameter state

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::AM
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
end

## Initialize AM state

sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::AM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
) =
  MuvAMState(
    generate_empty(pstate),
    tuner_state(parameter, sampler, tuner),
    copy(pstate.value),
    copy(sampler.C0),
    RealLowerTriangular(Array(eltype(pstate), pstate.size, pstate.size))
  )

## Reset parameter state

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
  sstate::MuvAMState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::AM,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.lastmean = copy(pstate.value)
  sstate.C = copy(sampler.C0)
  sstate.cholC = chol(sstate.C, Val{:L})
  sstate.count = 0
end

function covariance!(
  C::RealMatrix,
  lastC::RealMatrix,
  k::Integer,
  x::RealVector,
  lastmean::RealVector,
  secondlastmean::RealVector,
  d::Integer,
  sd::Real,
  ε::Real
)
  C[:, :] = zeros(d, d)
  BLAS.ger!(1.0, x, x, C)
  BLAS.ger!(1.0, -(k+1)*lastmean, lastmean, C)
  BLAS.ger!(1.0, k*secondlastmean, secondlastmean, C)
  C[:, :] = ((k-1)*lastC+sd*(C+ε*eye(d)))/k
end

Base.show(io::IO, sampler::AM) = print(io, "AM sampler: t0 = $(sampler.t0), sd = $(sampler.sd), ε = $(sampler.ε)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::AM) = show(io, sampler)
