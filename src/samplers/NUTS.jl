abstract NUTSState{F<:VariateForm} <: HMCSamplerState{F}

type MuvNUTSState <: NUTSState{Multivariate}
  pstateplus::ParameterState{Continuous, Multivariate}
  pstateminus::ParameterState{Continuous, Multivariate}
  pstateprime::ParameterState{Continuous, Multivariate}
  pstatedprime::ParameterState{Continuous, Multivariate}
  tune::MCTunerState
  ratio::Real
  a::Real
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

  function MuvNUTSState(
    pstateplus::ParameterState{Continuous, Multivariate},
    pstateminus::ParameterState{Continuous, Multivariate},
    pstateprime::ParameterState{Continuous, Multivariate},
    pstatedprime::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    ratio::Real,
    a::Real,
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
    count::Integer
  )
    if !isnan(ratio)
      @assert ratio > 0 "Acceptance ratio should be positive"
    end
    if !isnan(a)
      @assert 0 <= a <= 1 "Acceptance probability should be in [0, 1]"
    end
    @assert count >= 0 "Number of iterations (count) should be non-negative"
    new(
      pstateplus,
      pstateminus,
      pstateprime,
      pstatedprime,
      tune,
      ratio,
      a,
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
      count
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
    NaN,
    Array(eltype(pstate), pstate.size),
    Array(eltype(pstate), pstate.size),
    Array(eltype(pstate), pstate.size),
    Array(eltype(pstate), pstate.size),
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
    0
  )

immutable NUTS <: HMCSampler
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

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::NUTS
)
  parameter.uptogradlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
end

sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::NUTS,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
) =
  MuvNUTSState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))

uturn(positionplus::RealVector, positionminus::RealVector, momentumplus::RealVector, momentumminus::RealVector) =
  dot(positionplus-positionminus, momentumplus) < 0. || dot(positionplus-positionminus, momentumminus) < 0.

function build_tree!(
  sstate::NUTSState{Multivariate},
  pstate::ParameterState{Continuous, Multivariate},
  momentum::RealVector,
  u::Real,
  v::Real,
  j::Integer,
  step::Real,
  logtarget!::Function,
  gradlogtarget!::Function,
  sampler::NUTS
)
  if j == 0 # Base case: take one leapfrog step in the direction v
    local hamiltonianprime::Real

    leapfrog!(sstate.pstateprime, sstate.momentumprime, pstate, momentum, v*sstate.tune.step, gradlogtarget!)
    logtarget!(sstate.pstateprime)
    hamiltonianprime = hamiltonian(sstate.pstateprime.logtarget, sstate.momentumprime)

    sstate.nprime = Int(u <= hamiltonianprime)
    sstate.sprime = u < sampler.maxδ+hamiltonianprime

    return sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.momentumprime,
      sstate.pstateprime,
      sstate.nprime,
      sstate.sprime
  else # Recursion: implicitly build the left and right subtrees
    sstate.pstateminus,
    sstate.momentumminus,
    sstate.pstateplus,
    sstate.momentumplus,
    sstate.pstateprime,
    sstate.nprime,
    sstate.sprime =
      build_tree!(sstate, pstate, momentum, u, v, j-1, step, logtarget!, gradlogtarget!, sampler)

    if sstate.sprime
      if v == -1
        sstate.pstateminus, sstate.momentumminus, _, _, sstate.pstatedprime, sstate.ndprime, sstate.sdprime =
          build_tree!(
            sstate,
            sstate.pstateminus,
            sstate.momentumminus,
            u,
            v,
            j-1,
            step,
            logtarget!,
            gradlogtarget!,
            sampler
          )
      else
        _, _, sstate.pstateplus, sstate.momentumplus, sstate.pstatedprime, sstate.ndprime, sstate.sdprime =
          build_tree!(
            sstate,
            sstate.pstateplus,
            sstate.momentumplus,
            u,
            v,
            j-1,
            step,
            logtarget!,
            gradlogtarget!,
            sampler
          )
      end

      if rand() <= sstate.ndprime/(sstate.ndprime+sstate.nprime)
        sstate.pstateprime.value = copy(sstate.pstatedprime.value)
        sstate.pstateprime.gradlogtarget = copy(sstate.pstatedprime.gradlogtarget)
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
