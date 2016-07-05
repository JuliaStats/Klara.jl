### Abstract AMWG state

abstract AMWGState <: MCSamplerState

### AMWG state subtypes

# UnvAMWGState holds the internal state ("local variables") of the AMWG sampler for univariate parameters

type UnvAMWGState <: AMWGState
  proposal::ContinuousUnivariateDistribution
  pstate::ParameterState{Continuous, Univariate}
  tune::UnvRobertsRosenthalMCTune
  ratio::Real # Acceptance ratio

  function UnvAMWGState(
    proposal::ContinuousUnivariateDistribution,
    pstate::ParameterState{Continuous, Univariate},
    tune::UnvRobertsRosenthalMCTune,
    ratio::Real
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
    end
    new(proposal, pstate, tune, ratio)
  end
end

UnvAMWGState(
  proposal::ContinuousUnivariateDistribution,
  pstate::ParameterState{Continuous, Univariate},
  tune::UnvRobertsRosenthalMCTune=UnvRobertsRosenthalMCTune()
) =
  UnvAMWGState(proposal, pstate, tune, NaN)

# MuvAMWGState holds the internal state ("local variables") of the AMWG sampler for multivariate parameters

type MuvAMWGState <: AMWGState
  proposal::Vector{ContinuousUnivariateDistribution}
  pstate::ParameterState{Continuous, Multivariate}
  tune::MuvRobertsRosenthalMCTune
  ratio::Real # Acceptance ratio

  function MuvAMWGState(
    proposal::Vector{ContinuousUnivariateDistribution},
    pstate::ParameterState{Continuous, Multivariate},
    tune::MuvRobertsRosenthalMCTune,
    ratio::Real
  )
    for r in ratio
      if !isnan(r)
        @assert 0 < r < 1 "Acceptance ratio should be between 0 and 1"
      end
    end
    new(proposal, pstate, tune, ratio)
  end
end

MuvAMWGState(
  proposal::Vector{ContinuousUnivariateDistribution},
  pstate::ParameterState{Continuous, Multivariate},
  tune::MuvRobertsRosenthalMCTune
) =
  MuvAMWGState(proposal, pstate, tune, NaN)

immutable AMWG <: MHSampler
  σ0::RealVector
  lower::Real
  upper::Real
  symmetric::Bool # If symmetric=true then the proposal distribution is symmetric, else it is asymmetric
  setproposal::Function # Function for setting the proposal distribution
end

function AMWG(σ0::RealVector, lower::Real=-Inf, upper::Real=Inf)
  symmetric, setproposal =
    if lower == -Inf && upper == Inf
      true, (x, σ) -> Normal(x, σ)
    else
      false, (x, σ) -> Truncated(Normal(x, σ), lower, upper)
    end

  AMWG(σ0, lower, upper, symmetric, setproposal)
end
