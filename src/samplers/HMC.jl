### Abstract HMC state

abstract type HMCState{F<:VariateForm} <: HMCSamplerState{F} end

### HMC state subtypes

## UnvHMCState holds the internal state ("local variables") of the HMC sampler for univariate parameters

mutable struct UnvHMCState <: HMCState{Univariate}
  pstate::ParameterState{Continuous, Univariate} # Parameter state used internally by HMC
  tune::MCTunerState
  nleaps::Integer
  ratio::Real
  a::Real
  momentum::Real
  oldhamiltonian::Real
  newhamiltonian::Real
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function UnvHMCState(
    pstate::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    nleaps::Integer,
    ratio::Real,
    a::Real,
    momentum::Real,
    oldhamiltonian::Real,
    newhamiltonian::Real,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(a)
      @assert 0 <= a <= 1 "Acceptance probability should be in [0, 1]"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(pstate, tune, nleaps, ratio, a, momentum, oldhamiltonian, newhamiltonian, count, diagnosticindices)
  end
end

UnvHMCState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune(), nleaps::Integer=0) =
  UnvHMCState(pstate, tune, nleaps, NaN, NaN, NaN, NaN, NaN, 0, Dict{Symbol, Integer}())

## MuvHMCState holds the internal state ("local variables") of the HMC sampler for multivariate parameters

mutable struct MuvHMCState <: HMCState{Multivariate}
  pstate::ParameterState{Continuous, Multivariate} # Parameter state used internally by HMC
  tune::MCTunerState
  nleaps::Integer
  ratio::Real
  a::Real
  momentum::RealVector
  oldhamiltonian::Real
  newhamiltonian::Real
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function MuvHMCState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    nleaps::Integer,
    ratio::Real,
    a::Real,
    momentum::RealVector,
    oldhamiltonian::Real,
    newhamiltonian::Real,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(a)
      @assert 0 <= a <= 1 "Acceptance probability should be in [0, 1]"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(pstate, tune, nleaps, ratio, a, momentum, oldhamiltonian, newhamiltonian, count, diagnosticindices)
  end
end

MuvHMCState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune(), nleaps::Integer=0) =
  MuvHMCState(pstate, tune, nleaps, NaN, NaN, Array{eltype(pstate)}(pstate.size), NaN, NaN, 0, Dict{Symbol, Integer}())

### Hamiltonian Monte Carlo (HMC)

struct HMC <: HMCSampler
  leapstep::Real
  nleaps::Integer

  function HMC(leapstep::Real, nleaps::Integer)
    @assert leapstep > 0 "Leapfrog step is not positive"
    @assert nleaps > 0 "Number of leapfrog steps is not positive"
    new(leapstep, nleaps)
  end
end

HMC(leapstep::Real=0.1) = HMC(leapstep, 10)

### Initialize HMC sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::HMC,
  outopts::Dict
) where F<:VariateForm
  parameter.uptogradlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite.(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

## Initialize HMC state

tuner_state(parameter::Parameter, sampler::HMC, tuner::DualAveragingMCTuner) =
  DualAveragingMCTune(
  step=sampler.leapstep,
  λ=sampler.nleaps*sampler.leapstep,
  εbar=tuner.ε0bar,
  hbar=tuner.h0bar,
  accepted=0,
  proposed=0,
  totproposed=tuner.period
)

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::HMC,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvHMCState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts),
    tuner_state(parameter, sampler, tuner),
    sampler.nleaps
  )
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMC,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvHMCState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts),
    tuner_state(parameter, sampler, tuner),
    sampler.nleaps
  )
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvHMCState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts),
    tuner_state(parameter, sampler, tuner),
    0
  )

  sstate.tune.step = initialize_step!(
    sstate.pstate, pstate, randn(), sstate.tune.step, parameter.gradlogtarget!, typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)

  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvHMCState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts),
    tuner_state(parameter, sampler, tuner),
    0
  )

  sstate.tune.step = initialize_step!(
    sstate.pstate, sstate.momentum, pstate, randn(pstate.size), sstate.tune.step, parameter.gradlogtarget!, typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  set_diagnosticindices!(sstate, [:accept], diagnostickeys)

  sstate
end

## Reset parameter state

function reset!(tune::DualAveragingMCTune, sampler::HMC, tuner::DualAveragingMCTuner)
  tune.step = 1
  tune.λ = sampler.nleaps*sampler.leapstep
  tune.εbar = tuner.ε0bar
  tune.hbar = tuner.h0bar
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

function reset!(
  sstate::MuvHMCState,
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.tune.step = initialize_step!(
    sstate.pstate, pstate, randn(), sstate.tune.step, parameter.gradlogtarget!, typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate.count = 0
end

function reset!(
  sstate::MuvHMCState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::HMC,
  tuner::DualAveragingMCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.tune.step = initialize_step!(
    sstate.pstate, sstate.momentum, pstate, randn(pstate.size), sstate.tune.step, parameter.gradlogtarget!, typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate.count = 0
end

leapfrog!(sstate::HMCState{Univariate}, gradlogtarget!::Function) =
  sstate.momentum = leapfrog!(sstate.pstate, sstate.pstate, sstate.momentum, sstate.tune.step, gradlogtarget!)

leapfrog!(sstate::HMCState{Multivariate}, gradlogtarget!::Function) =
  leapfrog!(sstate.pstate, sstate.momentum, sstate.pstate, sstate.momentum, sstate.tune.step, gradlogtarget!)

show(io::IO, sampler::HMC) =
  print(io, "HMC sampler: number of leaps = $(sampler.nleaps), leap step = $(sampler.leapstep)")
