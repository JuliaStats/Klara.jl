### RAM holds the fields of the robust adaptive Metropolis (RAM) sampler

immutable RAM <: MHSampler
  jumpscale::Float64 # Scale factor for adapting the jump size
  targetrate::Float64 # Target acceptance rate

  function RAM(s::Float64, r::Float64)
    @assert s > 0 "Scale factor for adapting the jump size is not positive."
    @assert 0 < r < 1 "Target acceptance rate is not between 0 and 1"
    new(s, r)
  end
end

RAM() = RAM(1., 0.234)
RAM(s::Float64) = RAM(s, 0.234)

RAM(; jumpscale::Float64=1.0, targetrate::Float64=0.234) = RAM(jumpscale, targetrate)

### RAMStash type holds the internal state ("local variables") of the RAM sampler

type RAMStash <: MCStash{MCBaseSample}
  instate::MCState{MCBaseSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{MCBaseSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  ratio::Float64
  eta::Float64
  SS::Matrix{Float64}
  S::Matrix{Float64}
  rnormal::Vector{Float64}
end

RAMStash() =
  RAMStash(MCState(MCBaseSample(), MCBaseSample()), MCState(MCBaseSample(), MCBaseSample()), VanillaMCTune(), 0, NaN,
  NaN, Array(Float64, 0, 0), Array(Float64, 0, 0), Float64[])

RAMStash(l::Int) =
  RAMStash(MCState(MCBaseSample(l), MCBaseSample(l)), MCState(MCBaseSample(l), MCBaseSample(l)), VanillaMCTune(), 0,
  NaN, NaN, fill(NaN, l, l), fill(NaN, l, l), fill(NaN, l))

### Initialize RAM sampler

function initialize(m::MCModel, s::RAM, r::MCRunner, t::MCTuner)
  stash::RAMStash = RAMStash(m.size)

  stash.instate.current = MCBaseSample(copy(m.init))
  logtarget!(stash.instate.current, m.eval)
  @assert isfinite(stash.instate.current.logtarget) "Initial values out of model support."

  stash.S = Float64[i == j ? (m.scale.*s.jumpscale)[i] : 0. for i in 1:m.size, j in 1:m.size]

  stash.tune = VanillaMCTune()

  stash.count = 1

  stash
end

function initialize_task(m::MCModel, s::RAM, r::MCRunner, t::MCTuner)
  stash::RAMStash = initialize(m, s, r, t)

  # Hook inside Task to allow remote resetting
  task_local_storage(:reset,
    (x::Vector{Float64}) ->
    (stash.instate.current = MCBaseSample(copy(x)); logtarget!(stash.instate.current, m.eval))) 

  while true
    iterate!(stash, m, s, r, t, produce)
  end
end

### Perform iteration for RAM sampler

function iterate!(stash::RAMStash, m::MCModel, s::RAM, r::MCRunner, t::MCTuner, send::Function)
  if t.verbose
    stash.tune.proposed += 1
  end

  stash.rnormal = randn(m.size)
  stash.instate.successive = MCBaseSample(stash.instate.current.sample+stash.S*stash.rnormal)
  logtarget!(stash.instate.successive, m.eval)

  stash.ratio = stash.instate.successive.logtarget-stash.instate.current.logtarget
  if stash.ratio > 0 || (stash.ratio > log(rand()))
    stash.outstate = MCState(stash.instate.successive, stash.instate.current,
    {"accept" => true, "scale" => trace(stash.S)})
    stash.instate.current = deepcopy(stash.instate.successive)

    if t.verbose
      stash.tune.accepted += 1
    end
  else
    stash.outstate = MCState(stash.instate.current, stash.instate.current,
    {"accept" => false, "scale" => trace(stash.S)})
  end

  if t.verbose && stash.count <= r.burnin && mod(stash.count, t.period) == 0
    rate!(stash.tune)
    println("Burnin iteration $(stash.count) of $(r.burnin): ", round(100*stash.tune.rate, 2), " % acceptance rate")
  end

  stash.eta = min(1, m.size*stash.count^(-2/3))

  stash.SS =
    (stash.rnormal*stash.rnormal')/dot(stash.rnormal, stash.rnormal)*stash.eta*(min(1, exp(stash.ratio))-s.targetrate)
  stash.SS = stash.S*(eye(m.size)+stash.SS)*stash.S'
  stash.S = chol(stash.SS)'

  stash.count += 1

  send(stash.outstate)
end
