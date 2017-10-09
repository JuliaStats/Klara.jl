mutable struct AMWGState{F<:VariateForm} <: MHSamplerState{F}
  pstate::ParameterState{Continuous, F}
  tune::RobertsRosenthalMCTune{F}
  ratio::Real # Acceptance ratio
  diagnosticindices::Dict{Symbol, Integer}

  function AMWGState{F}(
    pstate::ParameterState{Continuous, F},
    tune::RobertsRosenthalMCTune{F},
    ratio::Real,
    diagnosticindices::Dict{Symbol, Integer}
  ) where {F<:VariateForm}
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    new(pstate, tune, ratio, diagnosticindices)
  end
end

AMWGState(pstate::ParameterState{Continuous, F}, tune::RobertsRosenthalMCTune{F}, ratio::Real) where {F<:VariateForm} =
  AMWGState{F}(pstate, tune, ratio, Dict{Symbol, Integer}())

AMWGState(pstate::ParameterState{Continuous, F}, tune::RobertsRosenthalMCTune{F}) where {F<:VariateForm} =
  AMWGState(pstate, tune, NaN)
