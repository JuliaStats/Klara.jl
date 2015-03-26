### HMC holds the fields of the HMC sampler
### These fields represent the initial user-defined state of the sampler
### These sampler fields are copied to the corresponding heap type fields, where the latter can be tuned

immutable HMC <: HMCSampler
  nleaps::Int # Number of leapfrog steps
  leapstep::Float64 # Leapfrog stepsize

  function HMC(nl::Int, l::Float64)
    @assert nl > 0 "Number of leapfrog steps is not positive."
    @assert l > 0 "Leapfrog stepsize is not positive."
    new(nl, l)
  end
end

HMC(nl::Int) = HMC(nl, 0.1)
HMC(l::Float64) = HMC(10, l)

HMC(; nleaps::Int=10, leapstep::Float64=0.1) = HMC(nleaps, leapstep)

### Sample type for HMC

type HMCSample <: MCSample{FirstOrder}
  sample::Vector{Float64} # sample position, i.e. model parameters
  logtarget::Float64
  gradlogtarget::Vector{Float64}
  momentum::Vector{Float64}
  hamiltonian::Float64
end

HMCSample(s::Vector{Float64}) = HMCSample(s, NaN, Float64[], Float64[], NaN)
HMCSample() = HMCSample(Float64[], NaN, Float64[], Float64[], NaN)

HMCSample(l::Int) = HMCSample(fill(NaN, l), NaN, fill(NaN, l), fill(NaN, l), NaN)

hamiltonian!(s::HMCSample) = (s.hamiltonian = -s.logtarget+0.5*dot(s.momentum, s.momentum))

function leapfrog(s::HMCSample, f::Function, leapstep::Float64)
  lsample = deepcopy(s)
  lsample.momentum += 0.5*lsample.gradlogtarget*leapstep
  lsample.sample += leapstep*lsample.momentum
  gradlogtargetall!(lsample, f)
  lsample.momentum += 0.5*lsample.gradlogtarget*leapstep
  hamiltonian!(lsample)
  lsample
end

### HMCHeap type holds the internal state ("local variables") of the HMC sampler

type HMCHeap <: MCHeap{HMCSample}
  instate::MCState{HMCSample} # Monte Carlo state used internally by the sampler
  outstate::MCState{HMCSample} # Monte Carlo state outputted by the sampler
  tune::MCTune
  count::Int # Current number of iterations
  nleaps::Int # Number of leapfrog steps for the current Monte Carlo iteration (possibly tuned)
  leapstep::Float64 # Leapfrog stepsize for the current Monte Carlo iteration (possibly tuned)
end

HMCHeap() = HMCHeap(MCState(HMCSample(), HMCSample()), MCState(HMCSample(), HMCSample()), VanillaMCTune(), 0, 0, NaN)

HMCHeap(l::Int, t::MCTune=VanillaMCTune()) =
  HMCHeap(MCState(HMCSample(l), HMCSample(l)), MCState(HMCSample(l), HMCSample(l)), t, 0, 0, NaN)

### Initialize HMC sampler

function initialize_heap(m::MCModel, s::HMC, r::MCRunner, t::MCTuner)
  @assert hasgradient(m) "HMC sampler requires model with gradient function."
  heap::HMCHeap = HMCHeap(m.size)

  heap.instate.current = HMCSample(copy(m.init))
  gradlogtargetall!(heap.instate.current, m.evalallg)
  @assert isfinite(heap.instate.current.logtarget) "Initial values out of model support."

  if isa(t, VanillaMCTuner)
    heap.tune = VanillaMCTune()
  elseif isa(t, EmpiricalMCTuner)
    heap.tune = EmpiricalMCTune(s.leapstep, s.nleaps)
  end

  heap.count = 1

  heap
end

function reset!(heap::HMCHeap, x::Vector{Float64}, m::MCModel)
  heap.instate.current = HMCSample(copy(x))
  gradlogtargetall!(heap.instate.current, m.evalallg)
end

function initialize_task!(heap::HMCHeap, m::MCModel, s::HMC, r::MCRunner, t::MCTuner)
  # Hook inside Task to allow remote resetting
  task_local_storage(:reset, (x::Vector{Float64})->reset!(heap, x))

  while true
    iterate!(heap, m, s, r, t, produce)
  end
end

### Perform iteration for HMC sampler

function iterate!(heap::HMCHeap, m::MCModel, s::HMC, r::MCRunner, t::MCTuner, send::Function)
  if isa(t, VanillaMCTuner)
    heap.nleaps, heap.leapstep = s.nleaps, s.leapstep
    if t.verbose
      heap.tune.proposed += 1
    end
  elseif isa(t, EmpiricalMCTuner)
    heap.nleaps, heap.leapstep = heap.tune.nsteps, heap.tune.step
    heap.tune.proposed += 1
  end

  heap.instate.current.momentum = randn(m.size)
  hamiltonian!(heap.instate.current)
  heap.instate.successive = deepcopy(heap.instate.current)

  for j = 1:heap.nleaps
    heap.instate.successive = leapfrog(heap.instate.successive, m.evalallg, heap.leapstep)
  end

  if rand() < exp(heap.instate.current.hamiltonian-heap.instate.successive.hamiltonian)
    heap.outstate = MCState(heap.instate.successive, heap.instate.current, Dict{Any, Any}("accept" => true))
    heap.instate.current = deepcopy(heap.instate.successive)

    if isa(t, VanillaMCTuner) && t.verbose
      heap.tune.accepted += 1
    elseif isa(t, EmpiricalMCTuner)
      heap.tune.accepted += 1
    end
  else
    heap.outstate = MCState(heap.instate.current, heap.instate.current, Dict{Any, Any}("accept" => false))
  end

  if isa(t, VanillaMCTuner) && t.verbose && heap.count <= r.burnin && mod(heap.count, t.period) == 0
    rate!(heap.tune)
    println("Burnin iteration $(heap.count) of $(r.burnin): ", round(100*heap.tune.rate, 2), " % acceptance rate")
  elseif isa(t, EmpiricalMCTuner) && heap.count <= r.burnin && mod(heap.count, t.period) == 0
    adapt!(heap.tune, t)
    reset!(heap.tune)
    if t.verbose
      println("Burnin iteration $(heap.count) of $(r.burnin): ", round(100*heap.tune.rate, 2), " % acceptance rate")
    end
  end

  heap.count += 1

  send(heap.outstate)
end
