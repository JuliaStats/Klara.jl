### Reference:
### Gareth O. Roberts and Jeffrey S. Rosenthal
### Examples of Adaptive MCMC
### Journal of Computational and Graphical Statistics, 2009, 18 (2), pp 349-367

### AMWG state

abstract type AMWGState{F<:VariateForm} <: MHSamplerState{F} end

mutable struct UnvAMWGState <: AMWGState{Univariate}
  proposal::Union{Distribution{Univariate, Continuous}, Void}
  pstate::ParameterState{Continuous, Univariate}
  tune::UnvRobertsRosenthalMCTune
  ratio::Real
  accept::Bool
  diagnosticindices::Dict{Symbol, Integer}

  function UnvAMWGState(
    proposal::Union{Distribution{Univariate, Continuous}, Void},
    pstate::ParameterState{Continuous, Univariate},
    tune::UnvRobertsRosenthalMCTune,
    ratio::Real,
    accept::Bool,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(proposal, pstate, tune, ratio, accept, diagnosticindices)
  end
end

UnvAMWGState(
  proposal::Union{Distribution{Univariate, Continuous}, Void},
  pstate::ParameterState{Continuous, Univariate},
  tune::UnvRobertsRosenthalMCTune,
) =
  UnvAMWGState(proposal, pstate, tune, NaN, true, Dict{Symbol, Integer}())

mutable struct MuvAMWGState <: AMWGState{Multivariate}
  proposal::Union{Distribution{Univariate, Continuous}, Void}
  pstate::ParameterState{Continuous, Multivariate}
  tune::MuvRobertsRosenthalMCTune
  ratio::Real
  accept::Vector{Bool}
  diagnosticindices::Dict{Symbol, Integer}

  function MuvAMWGState(
    proposal::Union{Distribution{Univariate, Continuous}, Void},
    pstate::ParameterState{Continuous, Multivariate},
    tune::MuvRobertsRosenthalMCTune,
    ratio::Real,
    accept::Vector{Bool},
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(proposal, pstate, tune, ratio, accept, diagnosticindices)
  end
end

MuvAMWGState(
  proposal::Union{Distribution{Univariate, Continuous}, Void},
  pstate::ParameterState{Continuous, Multivariate},
  tune::MuvRobertsRosenthalMCTune
) =
  MuvAMWGState(proposal, pstate, tune, NaN, Array{Bool}(pstate.size), Dict{Symbol, Integer}())

### Adaptive Metropolis-within-Gibbs (AMWG) sampler

abstract type AMWG <: MHSampler end

struct UnvAMWG <: AMWG
  lower::Real
  upper::Real
  logσ0::Real
end

UnvAMWG(; lower::Real=-Inf, upper::Real=Inf, σ::Real=1.) = UnvAMWG(lower, upper, log(σ))

function initialize!(
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::UnvAMWG,
  outopts::Dict
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

tuner_state(parameter::Parameter{Continuous, Univariate}, sampler::UnvAMWG, tuner::RobertsRosenthalMCTuner) =
  UnvRobertsRosenthalMCTune(logσ=sampler.logσ0, accepted=0, proposed=0, totproposed=tuner.period)

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::UnvAMWG,
  tuner::RobertsRosenthalMCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvAMWGState(nothing, generate_empty(pstate), tuner_state(parameter, sampler, tuner))
  set_diagnosticindices!(sstate, [:accept, :logσ], diagnostickeys)
  sstate
end

function reset!(
  pstate::ParameterState{Continuous, Univariate},
  x::Real,
  parameter::Parameter{Continuous, Univariate},
  sampler::UnvAMWG
)
  pstate.value = x
  parameter.logtarget!(pstate)
end

function reset!(tune::UnvRobertsRosenthalMCTune, sampler::UnvAMWG, tuner::RobertsRosenthalMCTuner)
  tune.logσ = sampler.logσ0
  tune.δ = NaN
  tune.batch = 0
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

reset!(
  sstate::UnvAMWGState,
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::UnvAMWG,
  tuner::RobertsRosenthalMCTuner
) =
  reset!(sstate.tune, sampler, tuner)

struct MuvAMWG <: AMWG
  lower::RealVector
  upper::RealVector
  logσ0::RealVector
end

function MuvAMWG(σ::RealVector)
  d = length(σ)
  MuvAMWG(fill(-Inf, d), fill(Inf, d), map(log, σ))
end

MuvAMWG(d::Integer; lower::RealVector=fill(-Inf, d), upper::RealVector=fill(Inf, d), σ::Real=1.) =
  MuvAMWG(lower, upper, fill(log(σ), d))

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MuvAMWG,
  outopts::Dict
)
  parameter.logtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array{Any}(length(pstate.diagnostickeys))
  end
end

tuner_state(parameter::Parameter{Continuous, Multivariate}, sampler::MuvAMWG, tuner::RobertsRosenthalMCTuner) =
  MuvRobertsRosenthalMCTune(sampler.logσ0, accepted=fill(0, length(sampler.logσ0)), proposed=0, totproposed=tuner.period)

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::MuvAMWG,
  tuner::RobertsRosenthalMCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvAMWGState(nothing, generate_empty(pstate), tuner_state(parameter, sampler, tuner))
  set_diagnosticindices!(sstate, [:accept, :logσ], diagnostickeys)
  sstate
end

function initialize_diagnosticvalues!(pstate::ParameterState{Continuous, Multivariate}, sstate::MuvAMWGState)
  if haskey(sstate.diagnosticindices, :accept)
    pstate.diagnosticvalues[sstate.diagnosticindices[:accept]] = Array{Bool}(pstate.size)
  end

  if haskey(sstate.diagnosticindices, :logσ)
    pstate.diagnosticvalues[sstate.diagnosticindices[:logσ]] = Array{Real}(pstate.size)
  end
end

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::MuvAMWG
)
  pstate.value = copy(x)
  parameter.logtarget!(pstate)
end

function reset!(tune::MuvRobertsRosenthalMCTune, sampler::MuvAMWG, tuner::RobertsRosenthalMCTuner)
  d = length(sampler.logσ0)
  tune.logσ = copy(sampler.logσ0)
  tune.δ = NaN
  tune.batch = 0
  tune.accepted = fill(0, d)
  tune.proposed = 0
  tune.totproposed = 0
  tune.rate = fill(NaN, d)
end

reset!(
  sstate::MuvAMWGState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MuvAMWG,
  tuner::RobertsRosenthalMCTuner
) =
  reset!(sstate.tune, sampler, tuner)
