abstract type NUTSState{F<:VariateForm} <: HMCSamplerState{F} end

mutable struct UnvNUTSState <: NUTSState{Univariate}
  pstateplus::ParameterState{Continuous, Univariate}
  pstateminus::ParameterState{Continuous, Univariate}
  pstateprime::ParameterState{Continuous, Univariate}
  pstatedprime::ParameterState{Continuous, Univariate}
  tune::MCTunerState
  ratio::Real
  momentum::Real
  momentumplus::Real
  momentumminus::Real
  momentumprime::Real
  oldhamiltonian::Real
  newhamiltonian::Real
  u::Real
  v::Integer
  j::Integer
  n::Integer
  nprime::Integer
  ndprime::Integer
  s::Bool
  sprime::Bool
  sdprime::Bool
  update::Bool
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function UnvNUTSState(
    pstateplus::ParameterState{Continuous, Univariate},
    pstateminus::ParameterState{Continuous, Univariate},
    pstateprime::ParameterState{Continuous, Univariate},
    pstatedprime::ParameterState{Continuous, Univariate},
    tune::MCTunerState,
    ratio::Real,
    momentum::Real,
    momentumplus::Real,
    momentumminus::Real,
    momentumprime::Real,
    oldhamiltonian::Real,
    newhamiltonian::Real,
    u::Real,
    v::Integer,
    j::Integer,
    n::Integer,
    nprime::Integer,
    ndprime::Integer,
    s::Bool,
    sprime::Bool,
    sdprime::Bool,
    update::Bool,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(
      pstateplus,
      pstateminus,
      pstateprime,
      pstatedprime,
      tune,
      ratio,
      momentum,
      momentumplus,
      momentumminus,
      momentumprime,
      oldhamiltonian,
      newhamiltonian,
      u,
      v,
      j,
      n,
      nprime,
      ndprime,
      s,
      sprime,
      sdprime,
      update,
      count,
      diagnosticindices
    )
  end
end

UnvNUTSState(pstate::ParameterState{Continuous, Univariate}, tune::MCTunerState=BasicMCTune()) =
  UnvNUTSState(
    pstate,
    pstate,
    pstate,
    pstate,
    tune,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    0,
    0,
    0,
    0,
    0,
    true,
    true,
    true,
    true,
    0,
    Dict{Symbol, Integer}()
  )

mutable struct MuvNUTSState <: NUTSState{Multivariate}
  pstateplus::ParameterState{Continuous, Multivariate}
  pstateminus::ParameterState{Continuous, Multivariate}
  pstateprime::ParameterState{Continuous, Multivariate}
  pstatedprime::ParameterState{Continuous, Multivariate}
  tune::MCTunerState
  ratio::Real
  momentum::RealVector
  momentumplus::RealVector
  momentumminus::RealVector
  momentumprime::RealVector
  oldhamiltonian::Real
  newhamiltonian::Real
  u::Real
  v::Integer
  j::Integer
  n::Integer
  nprime::Integer
  ndprime::Integer
  s::Bool
  sprime::Bool
  sdprime::Bool
  update::Bool
  count::Integer
  diagnosticindices::Dict{Symbol, Integer}

  function MuvNUTSState(
    pstateplus::ParameterState{Continuous, Multivariate},
    pstateminus::ParameterState{Continuous, Multivariate},
    pstateprime::ParameterState{Continuous, Multivariate},
    pstatedprime::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    momentum::RealVector,
    momentumplus::RealVector,
    momentumminus::RealVector,
    momentumprime::RealVector,
    oldhamiltonian::Real,
    newhamiltonian::Real,
    u::Real,
    v::Integer,
    j::Integer,
    n::Integer,
    nprime::Integer,
    ndprime::Integer,
    s::Bool,
    sprime::Bool,
    sdprime::Bool,
    update::Bool,
    count::Integer,
    diagnosticindices::Dict{Symbol, Integer}
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(
      pstateplus,
      pstateminus,
      pstateprime,
      pstatedprime,
      tune,
      ratio,
      momentum,
      momentumplus,
      momentumminus,
      momentumprime,
      oldhamiltonian,
      newhamiltonian,
      u,
      v,
      j,
      n,
      nprime,
      ndprime,
      s,
      sprime,
      sdprime,
      update,
      count,
      diagnosticindices
    )
  end
end

MuvNUTSState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvNUTSState(
    pstate,
    pstate,
    pstate,
    pstate,
    tune,
    NaN,
    Array{eltype(pstate)}(pstate.size),
    Array{eltype(pstate)}(pstate.size),
    Array{eltype(pstate)}(pstate.size),
    Array{eltype(pstate)}(pstate.size),
    NaN,
    NaN,
    NaN,
    0,
    0,
    0,
    0,
    0,
    true,
    true,
    true,
    true,
    0,
    Dict{Symbol, Integer}()
  )

mutable struct NUTS <: HMCSampler
  leapstep::Real
  maxδ::Integer
  maxndoublings::Integer

  function NUTS(leapstep::Real, maxδ::Integer, maxndoublings::Integer)
    @assert leapstep > 0 "Leapfrog step is not positive"
    @assert maxδ > 0 "maxδ is not positive"
    @assert maxndoublings > 0 "Maximum number of doublings is not positive"
    new(leapstep, maxδ, maxndoublings)
  end
end

NUTS(leapstep::Real=0.1; maxδ::Integer=1000, maxndoublings::Integer=5) = NUTS(leapstep, maxδ, maxndoublings)

function initialize!(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::NUTS,
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

tuner_state(parameter::Parameter, sampler::NUTS, tuner::DualAveragingMCTuner) =
  DualAveragingMCTune(
  step=sampler.leapstep,
  λ=NaN,
  εbar=tuner.ε0bar,
  hbar=tuner.h0bar,
  accepted=0,
  proposed=0,
  totproposed=tuner.period
)

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::NUTS,
  tuner::VanillaMCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvNUTSState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )
  set_diagnosticindices!(sstate, [:accept, :ndoublings], diagnostickeys)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::NUTS,
  tuner::VanillaMCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvNUTSState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )
  set_diagnosticindices!(sstate, [:accept, :ndoublings], diagnostickeys)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::NUTS,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = UnvNUTSState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )

  sstate.tune.step = initialize_step!(
    sstate.pstateplus, pstate, randn(), sstate.tune.step, parameter.gradlogtarget!, typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  set_diagnosticindices!(sstate, [:accept, :ndoublings, :a, :na], diagnostickeys)

  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::NUTS,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector,
  diagnostickeys::Vector{Symbol}
)
  sstate = MuvNUTSState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(parameter, sampler, tuner)
  )

  sstate.tune.step = initialize_step!(
    sstate.pstateplus,
    sstate.momentum,
    pstate,
    randn(pstate.size),
    sstate.tune.step,
    parameter.gradlogtarget!,
    typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  set_diagnosticindices!(sstate, [:accept, :ndoublings, :a, :na], diagnostickeys)

  sstate
end

function reset!(tune::DualAveragingMCTune, sampler::NUTS, tuner::DualAveragingMCTuner)
  tune.step = 1
  tune.εbar = tuner.ε0bar
  tune.hbar = tuner.h0bar
  (tune.accepted, tune.proposed, tune.totproposed, tune.rate) = (0, 0, tuner.period, NaN)
end

function reset!(
  sstate::MuvHMCState,
  pstate::ParameterState{Continuous, Univariate},
  parameter::Parameter{Continuous, Univariate},
  sampler::NUTS,
  tuner::DualAveragingMCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.tune.step = initialize_step!(
    sstate.pstateplus, pstate, randn(), sstate.tune.step, parameter.gradlogtarget!, typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate.count = 0
end

function reset!(
  sstate::MuvHMCState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::NUTS,
  tuner::DualAveragingMCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.tune.step = initialize_step!(
    sstate.pstateplus,
    sstate.momentum,
    pstate,
    randn(pstate.size),
    sstate.tune.step,
    parameter.gradlogtarget!,
    typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate.count = 0
end

uturn(positionplus::Real, positionminus::Real, momentumplus::Real, momentumminus::Real) =
  (positionplus-positionminus)*momentumplus < 0. || (positionplus-positionminus)*momentumminus < 0.

uturn(positionplus::RealVector, positionminus::RealVector, momentumplus::RealVector, momentumminus::RealVector) =
  dot(positionplus-positionminus, momentumplus) < 0. || dot(positionplus-positionminus, momentumminus) < 0.

function build_tree!(
  sstate::UnvNUTSState,
  pstate::ParameterState{Continuous, Univariate},
  momentum::Real,
  u::Real,
  v::Real,
  j::Integer,
  step::Real,
  parameter::Parameter{Continuous, Univariate},
  maxδ::Integer,
  ::Type{NUTS},
  ::Type{VanillaMCTuner},
  outopts::Dict
)
  if j == 0
    local hamiltonianprime::Real

    sstate.momentumprime = leapfrog!(sstate.pstateprime, pstate, momentum, v*sstate.tune.step, parameter.gradlogtarget!)

    parameter.logtarget!(sstate.pstateprime)
    hamiltonianprime = hamiltonian(sstate.pstateprime.logtarget, sstate.momentumprime)

    sstate.nprime = Int(u <= hamiltonianprime)
    sstate.sprime = u < maxδ+hamiltonianprime

    return sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime
  else
    (
      sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime
    ) =
      build_tree!(sstate, pstate, momentum, u, v, j-1, step, parameter, maxδ, NUTS, VanillaMCTuner, outopts)

    if sstate.sprime
      if v == -1
        (sstate.pstateminus, sstate.momentumminus, _, _, sstate.pstatedprime, sstate.ndprime, sstate.sdprime) =
          build_tree!(
            sstate,
            sstate.pstateminus,
            sstate.momentumminus,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            VanillaMCTuner,
            outopts
          )
      else
        (_, _, sstate.pstateplus, sstate.momentumplus, sstate.pstatedprime, sstate.ndprime, sstate.sdprime) =
          build_tree!(
            sstate,
            sstate.pstateplus,
            sstate.momentumplus,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            VanillaMCTuner,
            outopts
          )
      end

      if (rand() <= sstate.ndprime/(sstate.ndprime+sstate.nprime))
        sstate.pstateprime.value = sstate.pstatedprime.value

        sstate.pstateprime.gradlogtarget = sstate.pstatedprime.gradlogtarget
        if in(:gradloglikelihood, outopts[:monitor]) && parameter.gradloglikelihood! != nothing
          sstate.pstateprime.gradloglikelihood = sstate.pstatedprime.gradloglikelihood
        end
        if in(:gradlogprior, outopts[:monitor]) && parameter.gradlogprior! != nothing
          sstate.pstateprime.gradlogprior = sstate.pstatedprime.gradlogprior
        end

        sstate.pstateprime.logtarget = sstate.pstatedprime.logtarget
        if in(:loglikelihood, outopts[:monitor]) && parameter.loglikelihood! != nothing
          sstate.pstateprime.loglikelihood = sstate.pstatedprime.loglikelihood
        end
        if in(:logprior, outopts[:monitor]) && parameter.logprior! != nothing
          sstate.pstateprime.logprior = sstate.pstatedprime.logprior
        end
      end

      sstate.nprime += sstate.ndprime

      sstate.sprime =
        sstate.sdprime &&
        !uturn(sstate.pstateplus.value, sstate.pstateminus.value, sstate.momentumplus, sstate.momentumminus)
    end

    return sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime
  end
end

function build_tree!(
  sstate::MuvNUTSState,
  pstate::ParameterState{Continuous, Multivariate},
  momentum::RealVector,
  u::Real,
  v::Real,
  j::Integer,
  step::Real,
  parameter::Parameter{Continuous, Multivariate},
  maxδ::Integer,
  ::Type{NUTS},
  ::Type{VanillaMCTuner},
  outopts::Dict
)
  if j == 0
    local hamiltonianprime::Real

    leapfrog!(sstate.pstateprime, sstate.momentumprime, pstate, momentum, v*sstate.tune.step, parameter.gradlogtarget!)

    parameter.logtarget!(sstate.pstateprime)
    hamiltonianprime = hamiltonian(sstate.pstateprime.logtarget, sstate.momentumprime)

    sstate.nprime = Int(u <= hamiltonianprime)
    sstate.sprime = u < maxδ+hamiltonianprime

    return sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime
  else
    (
      sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime
    ) =
      build_tree!(sstate, pstate, momentum, u, v, j-1, step, parameter, maxδ, NUTS, VanillaMCTuner, outopts)

    if sstate.sprime
      if v == -1
        (sstate.pstateminus, sstate.momentumminus, _, _, sstate.pstatedprime, sstate.ndprime, sstate.sdprime) =
          build_tree!(
            sstate,
            sstate.pstateminus,
            sstate.momentumminus,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            VanillaMCTuner,
            outopts
          )
      else
        (_, _, sstate.pstateplus, sstate.momentumplus, sstate.pstatedprime, sstate.ndprime, sstate.sdprime) =
          build_tree!(
            sstate,
            sstate.pstateplus,
            sstate.momentumplus,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            VanillaMCTuner,
            outopts
          )
      end

      if (rand() <= sstate.ndprime/(sstate.ndprime+sstate.nprime))
        sstate.pstateprime.value = copy(sstate.pstatedprime.value)

        sstate.pstateprime.gradlogtarget = copy(sstate.pstatedprime.gradlogtarget)
        if in(:gradloglikelihood, outopts[:monitor]) && parameter.gradloglikelihood! != nothing
          sstate.pstateprime.gradloglikelihood = copy(sstate.pstatedprime.gradloglikelihood)
        end
        if in(:gradlogprior, outopts[:monitor]) && parameter.gradlogprior! != nothing
          sstate.pstateprime.gradlogprior = copy(sstate.pstatedprime.gradlogprior)
        end

        sstate.pstateprime.logtarget = sstate.pstatedprime.logtarget
        if in(:loglikelihood, outopts[:monitor]) && parameter.loglikelihood! != nothing
          sstate.pstateprime.loglikelihood = sstate.pstatedprime.loglikelihood
        end
        if in(:logprior, outopts[:monitor]) && parameter.logprior! != nothing
          sstate.pstateprime.logprior = sstate.pstatedprime.logprior
        end
      end

      sstate.nprime += sstate.ndprime

      sstate.sprime =
        sstate.sdprime &&
        !uturn(sstate.pstateplus.value, sstate.pstateminus.value, sstate.momentumplus, sstate.momentumminus)
    end

    return sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime
  end
end

function build_tree!(
  sstate::UnvNUTSState,
  pstate::ParameterState{Continuous, Univariate},
  momentum::Real,
  oldhamiltonian::Real,
  u::Real,
  v::Real,
  j::Integer,
  step::Real,
  parameter::Parameter{Continuous, Univariate},
  maxδ::Integer,
  ::Type{NUTS},
  ::Type{DualAveragingMCTuner},
  outopts::Dict
)
  local aprime::Real
  local adprime::Real
  local naprime::Integer
  local nadprime::Integer

  if j == 0
    local hamiltonianprime::Real

    sstate.momentumprime = leapfrog!(sstate.pstateprime, pstate, momentum, v*sstate.tune.step, parameter.gradlogtarget!)

    parameter.logtarget!(sstate.pstateprime)
    hamiltonianprime = hamiltonian(sstate.pstateprime.logtarget, sstate.momentumprime)

    sstate.nprime = Int(u <= hamiltonianprime)
    sstate.sprime = u < maxδ+hamiltonianprime

    return sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime,
      min(1, exp(hamiltonianprime-oldhamiltonian)),
      1
  else
    (
      sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime,
      aprime,
      naprime
    ) =
      build_tree!(
        sstate, pstate, momentum, oldhamiltonian, u, v, j-1, step, parameter, maxδ, NUTS, DualAveragingMCTuner, outopts
      )

    if sstate.sprime
      if v == -1
        (
          sstate.pstateminus,
          sstate.momentumminus,
          _,
          _,
          sstate.pstatedprime,
          sstate.ndprime,
          sstate.sdprime,
          adprime,
          nadprime
        ) =
          build_tree!(
            sstate,
            sstate.pstateminus,
            sstate.momentumminus,
            oldhamiltonian,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            DualAveragingMCTuner,
            outopts
          )
      else
        (
          _,
          _,
          sstate.pstateplus,
          sstate.momentumplus,
          sstate.pstatedprime,
          sstate.ndprime,
          sstate.sdprime,
          adprime,
          nadprime
        ) =
          build_tree!(
            sstate,
            sstate.pstateplus,
            sstate.momentumplus,
            oldhamiltonian,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            DualAveragingMCTuner,
            outopts
          )
      end

      if (rand() <= sstate.ndprime/(sstate.ndprime+sstate.nprime))
        sstate.pstateprime.value = sstate.pstatedprime.value

        sstate.pstateprime.gradlogtarget = sstate.pstatedprime.gradlogtarget
        if in(:gradloglikelihood, outopts[:monitor]) && parameter.gradloglikelihood! != nothing
          sstate.pstateprime.gradloglikelihood = sstate.pstatedprime.gradloglikelihood
        end
        if in(:gradlogprior, outopts[:monitor]) && parameter.gradlogprior! != nothing
          sstate.pstateprime.gradlogprior = sstate.pstatedprime.gradlogprior
        end

        sstate.pstateprime.logtarget = sstate.pstatedprime.logtarget
        if in(:loglikelihood, outopts[:monitor]) && parameter.loglikelihood! != nothing
          sstate.pstateprime.loglikelihood = sstate.pstatedprime.loglikelihood
        end
        if in(:logprior, outopts[:monitor]) && parameter.logprior! != nothing
          sstate.pstateprime.logprior = sstate.pstatedprime.logprior
        end
      end

      sstate.nprime += sstate.ndprime

      sstate.sprime =
        sstate.sdprime &&
        !uturn(sstate.pstateplus.value, sstate.pstateminus.value, sstate.momentumplus, sstate.momentumminus)

      aprime += adprime
      naprime += nadprime
    end

    return sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime,
      aprime,
      naprime
  end
end

function build_tree!(
  sstate::MuvNUTSState,
  pstate::ParameterState{Continuous, Multivariate},
  momentum::RealVector,
  oldhamiltonian::Real,
  u::Real,
  v::Real,
  j::Integer,
  step::Real,
  parameter::Parameter{Continuous, Multivariate},
  maxδ::Integer,
  ::Type{NUTS},
  ::Type{DualAveragingMCTuner},
  outopts::Dict
)
  local aprime::Real
  local adprime::Real
  local naprime::Integer
  local nadprime::Integer

  if j == 0
    local hamiltonianprime::Real

    leapfrog!(sstate.pstateprime, sstate.momentumprime, pstate, momentum, v*sstate.tune.step, parameter.gradlogtarget!)

    parameter.logtarget!(sstate.pstateprime)
    hamiltonianprime = hamiltonian(sstate.pstateprime.logtarget, sstate.momentumprime)

    sstate.nprime = Int(u <= hamiltonianprime)
    sstate.sprime = u < maxδ+hamiltonianprime

    return sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime,
      min(1, exp(hamiltonianprime-oldhamiltonian)),
      1
  else
    (
      sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime,
      aprime,
      naprime
    ) =
      build_tree!(
        sstate, pstate, momentum, oldhamiltonian, u, v, j-1, step, parameter, maxδ, NUTS, DualAveragingMCTuner, outopts
      )

    if sstate.sprime
      if v == -1
        (
          sstate.pstateminus,
          sstate.momentumminus,
          _,
          _,
          sstate.pstatedprime,
          sstate.ndprime,
          sstate.sdprime,
          adprime,
          nadprime
        ) =
          build_tree!(
            sstate,
            sstate.pstateminus,
            sstate.momentumminus,
            oldhamiltonian,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            DualAveragingMCTuner,
            outopts
          )
      else
        (
          _,
          _,
          sstate.pstateplus,
          sstate.momentumplus,
          sstate.pstatedprime,
          sstate.ndprime,
          sstate.sdprime,
          adprime,
          nadprime
        ) =
          build_tree!(
            sstate,
            sstate.pstateplus,
            sstate.momentumplus,
            oldhamiltonian,
            u,
            v,
            j-1,
            step,
            parameter,
            maxδ,
            NUTS,
            DualAveragingMCTuner,
            outopts
          )
      end

      if (rand() <= sstate.ndprime/(sstate.ndprime+sstate.nprime))
        sstate.pstateprime.value = copy(sstate.pstatedprime.value)

        sstate.pstateprime.gradlogtarget = copy(sstate.pstatedprime.gradlogtarget)
        if in(:gradloglikelihood, outopts[:monitor]) && parameter.gradloglikelihood! != nothing
          sstate.pstateprime.gradloglikelihood = copy(sstate.pstatedprime.gradloglikelihood)
        end
        if in(:gradlogprior, outopts[:monitor]) && parameter.gradlogprior! != nothing
          sstate.pstateprime.gradlogprior = copy(sstate.pstatedprime.gradlogprior)
        end

        sstate.pstateprime.logtarget = sstate.pstatedprime.logtarget
        if in(:loglikelihood, outopts[:monitor]) && parameter.loglikelihood! != nothing
          sstate.pstateprime.loglikelihood = sstate.pstatedprime.loglikelihood
        end
        if in(:logprior, outopts[:monitor]) && parameter.logprior! != nothing
          sstate.pstateprime.logprior = sstate.pstatedprime.logprior
        end
      end

      sstate.nprime += sstate.ndprime

      sstate.sprime =
        sstate.sdprime &&
        !uturn(sstate.pstateplus.value, sstate.pstateminus.value, sstate.momentumplus, sstate.momentumminus)

      aprime += adprime
      naprime += nadprime
    end

    return sstate.pstateminus,
      sstate.momentumminus,
      sstate.pstateplus,
      sstate.momentumplus,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime,
      aprime,
      naprime
  end
end

show(io::IO, sampler::NUTS) =
  print(
    io,
    "NUTS sampler: leap step = ",
    sampler.leapstep,
    ", doubling threshold = ",
    sampler.maxδ,
    ", maximum number of doublings = ",
    sampler.maxndoublings
  )
