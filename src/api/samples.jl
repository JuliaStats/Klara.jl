### Generic MCBaseSample type used by most samplers

type MCBaseSample <: MCSample{NullOrder}
  sample::Vector{Float64}
  logtarget::Float64
end

MCBaseSample(s::Vector{Float64}) = MCBaseSample(s, NaN)
MCBaseSample() = MCBaseSample(Float64[], NaN)

MCBaseSample(l::Int) = MCBaseSample(fill(NaN, l), NaN)

logtarget!(s::MCSample, f::Function) = (s.logtarget = f(s.sample))

### MCGradSample is used by samplers that compute (up to) the gradient of the log-target (for ex HMC and MALA)

type MCGradSample <: MCSample{FirstOrder}
  sample::Vector{Float64}
  logtarget::Float64
  gradlogtarget::Vector{Float64}
end

MCGradSample(s::Vector{Float64}) = MCGradSample(s, NaN, Float64[])
MCGradSample() = MCGradSample(Float64[], NaN, Float64[])

MCGradSample(l::Int) = MCGradSample(fill(NaN, l), NaN, fill(NaN, l))

gradlogtargetall!(s::MCSample, f::Function) = ((s.logtarget, s.gradlogtarget) = f(s.sample))

### MCMCTensorSample is used by samplers that compute (up to) the tensor of the log-target (for ex SMMALA)

type MCTensorSample <: MCSample{SecondOrder}
  sample::Vector{Float64}
  logtarget::Float64
  gradlogtarget::Vector{Float64}
  tensorlogtarget::Matrix{Float64}
end

MCTensorSample(s::Vector{Float64}) = MCTensorSample(s, NaN, Float64[], Array(Float64, 0, 0))
MCTensorSample() = MCTensorSample(Float64[], NaN, Float64[], Array(Float64, 0, 0))

MCTensorSample(l::Int) = MCTensorSample(fill(NaN, l), NaN, fill(NaN, l), fill(NaN, l, l))

tensorlogtargetall!(s::MCSample, f::Function) = ((s.logtarget, s.gradlogtarget, s.tensorlogtarget) = f(s.sample))

### MCMCDTensorSample is used by samplers that compute (up to) the derivative of the tensor of the log-target (for ex
### RMHMC and PMALA)

type MCDTensorSample <: MCSample{ThirdOrder}
  sample::Vector{Float64}
  logtarget::Float64
  gradlogtarget::Vector{Float64}
  tensorlogtarget::Matrix{Float64}
  dtensorlogtarget::Array{Float64, 3}
end

MCDTensorSample(s::Vector{Float64}) = MCDTensorSample(s, NaN, Float64[], Array(Float64, 0, 0), Array(Float64, 0, 0, 0))
MCDTensorSample() = MCDTensorSample(Float64[], NaN, Float64[], Array(Float64, 0, 0), Array(Float64, 0, 0, 0))

MCDTensorSample(l::Int) = MCDTensorSample(fill(NaN, l), NaN, fill(NaN, l), fill(NaN, l, l), fill(NaN, l, l, l))

dtensorlogtargetall!(s::MCSample, f::Function) =
  ((s.logtarget, s.gradlogtarget, s.tensorlogtarget, s.dtensorlogtarget) = f(s.sample))
