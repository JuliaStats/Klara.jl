module MCMC

using Base.LinAlg.BLAS
using Distributions
using ReverseDiffSource

# import Base.*
import Base:
  show,
  run,
  select,
  mean,
  var,
  std
import Distributions:
  Bernoulli,
  Beta,
  Binomial,
  Cauchy,
  Gamma,
  Laplace,
  LogNormal,
  Normal,
  Uniform,
  Weibull,
  logpdf,
  logcdf,
  logccdf

export
  ### types
  MCModel,
  MCLikModel,
  MCSampler,
  MHSampler,
  MHBaseSampler,
  MH,
  RAMSampler,
  RAM,
  HMCSampler,
  HMCBaseSampler,
  HMC,
  RMHMCSampler,
  LMCSampler,
  LMCBaseSampler,
  MALA,
  SMMALA,
  PMALA,
  SliceSampler,
  MCRunner,
  SerialMCRunner,
  ParrallelMCRunner,
  SerialMCBaseRunner,
  SerialMC,
  MCTuner,
  VanillaMCTuner,
  EmpiricalMCTuner,
  MCTask,
  MCJob,
  PlainMCJob,
  TaskMCJob,
  MCSample,
  MCBaseSample,
  MCGradSample,
  MCTensorSample,
  MCDTensorSample,
  HMCSample,
  MCState,
  MCChain,
  MCTune,
  VanillaMCTune,
  EmpiricalMCTune,
  MCStash,
  HMCStash,
  ### functions
  model,
  hasgradient,
  hastensor,
  hasdtensor,
  logtarget!,
  gradlogtargetall!,
  tensorlogtargetall!,
  dtensorlogtargetall!,
  hamiltonian!,
  leapfrog,
  reset!,
  count!,
  rate!,
  adapt!,
  run,
  select,
  mean,
  mcvar,
  mcse,
  ess,
  actime,
  acceptance,
  describe,
  linearzv,
  quadraticzv

include("parsers/expr_funcs.jl")
include("parsers/modelparser.jl")
include("parsers/definitions/DistributionsExtensions.jl")
include("parsers/definitions/AccumulatorDerivRules.jl")
include("parsers/definitions/MCMCDerivRules.jl")
include("models/models.jl")
include("models/MCLikModel.jl")
include("samplers/samplers.jl")
# include("samplers/ARS.jl")
# include("samplers/SliceSampler.jl")
# include("samplers/MH.jl")
# include("samplers/RAM.jl")
# include("samplers/IMH.jl")
include("samplers/HMC.jl")
# include("samplers/HMCDA.jl")
# include("samplers/NUTS.jl")
# include("samplers/MALA.jl")
# include("samplers/SMMALA.jl")
# include("samplers/RMHMC.jl")
# include("samplers/PMALA.jl")
# include("runners/runners.jl")
include("runners/SerialMC.jl")
# include("runners/SerialTempMC.jl")
# include("runners/SeqMC.jl")
include("tuners/tuners.jl")
include("tuners/VanillaMCTuner.jl")
include("tuners/EmpiricalMCTuner.jl")
include("jobs/jobs.jl")
include("jobs/PlainMCJob.jl")
include("jobs/TaskMCJob.jl")
include("stats/filtering.jl")
include("stats/mean.jl")
include("stats/var.jl")
include("stats/ess.jl")
include("stats/summary.jl")
include("stats/zv.jl")

end
