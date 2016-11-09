abstract NUTSState{F<:VariateForm} <: HMCSamplerState{F}

type UnvNUTSState <: NUTSState{Univariate}
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
    count::Integer
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
      count
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
    0
  )

type MuvNUTSState <: NUTSState{Multivariate}
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
    count::Integer
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

type NUTS <: HMCSampler
  leapstep::Real
  maxδ::Integer
  maxndoublings::Integer
  buildtree!::Function

  function NUTS(leapstep::Real, maxδ::Integer, maxndoublings::Integer, buildtree!::Function)
    @assert leapstep > 0 "Leapfrog step is not positive"
    @assert maxδ > 0 "maxδ is not positive"
    @assert maxndoublings > 0 "Maximum number of doublings is not positive"
    new(leapstep, maxδ, maxndoublings, buildtree!)
  end
end

NUTS(leapstep::Real=0.1; maxδ::Integer=1000, maxndoublings::Integer=5) = NUTS(leapstep, maxδ, maxndoublings, ()->())

function initialize!{F<:VariateForm}(
  pstate::ParameterState{Continuous, F},
  parameter::Parameter{Continuous, F},
  sampler::NUTS,
  outopts::Dict
)
  parameter.uptogradlogtarget!(pstate)
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array(Any, length(pstate.diagnostickeys))
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

sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::NUTS,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
) =
  UnvNUTSState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))

sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::NUTS,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
) =
  MuvNUTSState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))

function sampler_state(
  parameter::Parameter{Continuous, Univariate},
  sampler::NUTS,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Univariate},
  vstate::VariableStateVector
)
  sstate = UnvNUTSState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))
  sstate.tune.step = initialize_step!(
    sstate.pstateplus, pstate, randn(), sstate.tune.step, parameter.gradlogtarget!, typeof(tuner)
  )
  sstate.tune.μ = log(10*sstate.tune.step)
  sstate
end

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::NUTS,
  tuner::DualAveragingMCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvNUTSState(generate_empty(pstate), tuner_state(parameter, sampler, tuner))
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

function codegen_tree_builder{T<:MCTuner}(
  parameter::ContinuousParameter,
  ::Type{NUTS},
  tunertype::Type{T},
  outopts::Dict
)
  ifjzero = []
  ifjnotzero = []
  ifsprime = []
  ifvminusone = []
  ifvnotminusone = []
  update = []
  body = []

  if tunertype == DualAveragingMCTuner
    push!(body, :(local aprime::Real))
    push!(body, :(local adprime::Real))
    push!(body, :(local naprime::Integer))
    push!(body, :(local nadprime::Integer))
  end

  vform = variate_form(parameter)
  if vform != Univariate && vform != Multivariate
    error("Only univariate or multivariate parameter states allowed in HMC code generation")
  end

  @gensym tree_builder

  push!(ifjzero, :(local hamiltonianprime::Real))

  if vform == Univariate
    push!(
      ifjzero,
      :(
        _sstate.momentumprime = leapfrog!(_sstate.pstateprime, _pstate, _momentum, _v*_sstate.tune.step, _gradlogtarget!)
      )
    )
  elseif vform == Multivariate
    push!(
      ifjzero,
      :(
        leapfrog!(_sstate.pstateprime, _sstate.momentumprime, _pstate, _momentum, _v*_sstate.tune.step, _gradlogtarget!)
      )
    )
  end
  push!(ifjzero, :(_logtarget!(_sstate.pstateprime)))
  push!(ifjzero, :(hamiltonianprime = hamiltonian(_sstate.pstateprime.logtarget, _sstate.momentumprime)))

  push!(ifjzero, :(_sstate.nprime = Int(_u <= hamiltonianprime)))
  push!(ifjzero, :(_sstate.sprime = _u < _sampler.maxδ+hamiltonianprime))

  if tunertype == DualAveragingMCTuner
    push!(ifjzero, :(
      return _sstate.pstateprime,
        _sstate.momentumprime,
        _sstate.pstateprime,
        _sstate.momentumprime,
        _sstate.pstateprime,
        _sstate.nprime,
        _sstate.sprime,
        min(1, exp(hamiltonianprime-_oldhamiltonian)),
        1
    ))
  else
    push!(ifjzero, :(
      return _sstate.pstateprime,
        _sstate.momentumprime,
        _sstate.pstateprime,
        _sstate.momentumprime,
        _sstate.pstateprime,
        _sstate.nprime,
        _sstate.sprime
    ))
  end

  if tunertype == DualAveragingMCTuner
    push!(ifjnotzero, :(
      (
        _sstate.pstateminus,
        _sstate.momentumminus,
        _sstate.pstateplus,
        _sstate.momentumplus,
        _sstate.pstateprime,
        _sstate.nprime,
        _sstate.sprime,
        aprime,
        naprime
      ) =
        $tree_builder(
          _sstate,
          _pstate,
          _momentum,
          _oldhamiltonian,
          _u,
          _v,
          _j-1,
          _step,
          _logtarget!,
          _gradlogtarget!,
          _sampler
        )
    ))

    push!(ifvminusone, :(
      (
        _sstate.pstateminus,
        _sstate.momentumminus,
        _,
        _,
        _sstate.pstatedprime,
        _sstate.ndprime,
        _sstate.sdprime,
        adprime,
        nadprime
      ) =
        $tree_builder(
          _sstate,
          _sstate.pstateminus,
          _sstate.momentumminus,
          _oldhamiltonian,
          _u,
          _v,
          _j-1,
          _step,
          _logtarget!,
          _gradlogtarget!,
          _sampler
        )
    ))

    push!(ifvnotminusone, :(
      (
        _,
        _,
        _sstate.pstateplus,
        _sstate.momentumplus,
        _sstate.pstatedprime,
        _sstate.ndprime,
        _sstate.sdprime,
        adprime,
        nadprime
      ) =
        $tree_builder(
          _sstate,
          _sstate.pstateplus,
          _sstate.momentumplus,
          _oldhamiltonian,
          _u,
          _v,
          _j-1,
          _step,
          _logtarget!,
          _gradlogtarget!,
          _sampler
        )
    ))
  else
    push!(ifjnotzero, :(
      (
        _sstate.pstateminus,
        _sstate.momentumminus,
        _sstate.pstateplus,
        _sstate.momentumplus,
        _sstate.pstateprime,
        _sstate.nprime,
        _sstate.sprime
      ) =
        $tree_builder(_sstate, _pstate, _momentum, _u, _v, _j-1, _step, _logtarget!, _gradlogtarget!, _sampler)
    ))

    push!(ifvminusone, :(
      (
        _sstate.pstateminus,
        _sstate.momentumminus,
        _,
        _,
        _sstate.pstatedprime,
        _sstate.ndprime,
        _sstate.sdprime
      ) =
        $tree_builder(
          _sstate,
          _sstate.pstateminus,
          _sstate.momentumminus,
          _u,
          _v,
          _j-1,
          _step,
          _logtarget!,
          _gradlogtarget!,
          _sampler
        )
    ))

    push!(ifvnotminusone, :(
      (
        _,
        _,
        _sstate.pstateplus,
        _sstate.momentumplus,
        _sstate.pstatedprime,
        _sstate.ndprime,
        _sstate.sdprime
      ) =
        $tree_builder(
          _sstate,
          _sstate.pstateplus,
          _sstate.momentumplus,
          _u,
          _v,
          _j-1,
          _step,
          _logtarget!,
          _gradlogtarget!,
          _sampler
        )
    ))
  end

  push!(ifsprime, Expr(:if, :(_v == -1), Expr(:block, ifvminusone...), ifvnotminusone...))

  if vform == Univariate
    push!(update, :(_sstate.pstateprime.value = _sstate.pstatedprime.value))
    push!(update, :(_sstate.pstateprime.gradlogtarget = _sstate.pstatedprime.gradlogtarget))
    if in(:gradloglikelihood, outopts[:monitor]) && parameter.gradloglikelihood! != nothing
      push!(update, :(_sstate.pstateprime.gradloglikelihood = _sstate.pstatedprime.gradloglikelihood))
    end
    if in(:gradlogprior, outopts[:monitor]) && parameter.gradlogprior! != nothing
      push!(update, :(_sstate.pstateprime.gradlogprior = _sstate.pstatedprime.gradlogprior))
    end
  else
    push!(update, :(_sstate.pstateprime.value = copy(_sstate.pstatedprime.value)))
    push!(update, :(_sstate.pstateprime.gradlogtarget = copy(_sstate.pstatedprime.gradlogtarget)))
    if in(:gradloglikelihood, outopts[:monitor]) && parameter.gradloglikelihood! != nothing
      push!(update, :(_sstate.pstateprime.gradloglikelihood = copy(_sstate.pstatedprime.gradloglikelihood)))
    end
    if in(:gradlogprior, outopts[:monitor]) && parameter.gradlogprior! != nothing
      push!(update, :(_sstate.pstateprime.gradlogprior = copy(_sstate.pstatedprime.gradlogprior)))
    end
  end
  push!(update, :(_sstate.pstateprime.logtarget = _sstate.pstatedprime.logtarget))
  if in(:loglikelihood, outopts[:monitor]) && parameter.loglikelihood! != nothing
    push!(update, :(_sstate.pstateprime.loglikelihood = _sstate.pstatedprime.loglikelihood))
  end
  if in(:logprior, outopts[:monitor]) && parameter.logprior! != nothing
    push!(update, :(_sstate.pstateprime.logprior = _sstate.pstatedprime.logprior))
  end

  push!(ifsprime, Expr(:if, :(rand() <= _sstate.ndprime/(_sstate.ndprime+_sstate.nprime)), Expr(:block, update...)))

  push!(ifsprime, :(_sstate.nprime += _sstate.ndprime))

  push!(
    ifsprime,
    :(
      _sstate.sprime =
        _sstate.sdprime &&
        !uturn(_sstate.pstateplus.value, _sstate.pstateminus.value, _sstate.momentumplus, _sstate.momentumminus)
    )
  )

  if tunertype == DualAveragingMCTuner
    push!(ifsprime, :(aprime += adprime))
    push!(ifsprime, :(naprime += nadprime))
  end

  push!(ifjnotzero, Expr(:if, :(_sstate.sprime), Expr(:block, ifsprime...)))

  if tunertype == DualAveragingMCTuner
    push!(ifjnotzero, :(
      return _sstate.pstateminus,
        _sstate.momentumminus,
        _sstate.pstateplus,
        _sstate.momentumplus,
        _sstate.pstateprime,
        _sstate.nprime,
        _sstate.sprime,
        aprime,
        naprime
    ))
  else
    push!(ifjnotzero, :(
      return _sstate.pstateminus,
        _sstate.momentumminus,
        _sstate.pstateplus,
        _sstate.momentumplus,
        _sstate.pstateprime,
        _sstate.nprime,
        _sstate.sprime
    ))
  end

  push!(body, Expr(:if, :(_j == 0), Expr(:block, ifjzero...), Expr(:block, ifjnotzero...)))

  args = [
    :(_sstate::NUTSState{$vform}),
    :(_pstate::ParameterState{Continuous, $vform}),
    :(_momentum::$(vform == Univariate ? Real : RealVector))
  ]

  if tunertype == DualAveragingMCTuner
    push!(args, :(_oldhamiltonian::Real))
  end

  args = [args; [
    :(_u::Real),
    :(_v::Real),
    :(_j::Integer),
    :(_step::Real),
    :(_logtarget!::Function),
    :(_gradlogtarget!::Function),
    :(_sampler::NUTS)
  ]]

  quote
    function $tree_builder($(args...))
      $(body...)
    end
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
