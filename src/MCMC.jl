module MCMC

using Base.LinAlg.BLAS
using Distributions
using StatsBase
using ReverseDiffSource

import Base:
  convert,
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
  Exponential,
  Gamma,
  Laplace,
  LogNormal,
  Normal,
  Poisson,
  TDist,
  Uniform,
  Weibull,
  logpdf,
  logcdf,
  logccdf

export
  ### types
  MCLikModel,
  MCChain,
  ARS,
  SliceSampler,
  MH,
  RAM,
  HMC,
  MALA,
  SMMALA,
  SerialMC,
  VanillaMCTuner,
  EmpiricalMCTuner,
  MCJob,
  PlainMCJob,
  TaskMCJob,
  ### functions
  model,
  run,
  resume,
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

include("api/input.jl")
include("api/output.jl")
include("parsers/expr_funcs.jl")
include("parsers/modelparser.jl")
include("parsers/definitions/DistributionsExtensions.jl")
include("parsers/definitions/AccumulatorDerivRules.jl")
include("parsers/definitions/MCMCDerivRules.jl")
include("parsers/expr_funcs.jl")
include("models/MCLikModel.jl")
include("models/models.jl")
include("samplers/ARS.jl")
include("samplers/SliceSampler.jl")
include("samplers/MH.jl")
include("samplers/RAM.jl")
# include("samplers/IMH.jl")
include("samplers/HMC.jl")
# include("samplers/HMCDA.jl")
# include("samplers/NUTS.jl")
include("samplers/MALA.jl")
include("samplers/SMMALA.jl")
# include("samplers/RMHMC.jl")
# include("samplers/PMALA.jl")
include("runners/SerialMC.jl")
# include("runners/SerialTempMC.jl")
# include("runners/SeqMC.jl")
include("tuners/VanillaMCTuner.jl")
include("tuners/EmpiricalMCTuner.jl")
include("jobs/PlainMCJob.jl")
include("jobs/TaskMCJob.jl")
include("jobs/jobs.jl")
include("stats/filtering.jl")
include("stats/mean.jl")
include("stats/var.jl")
include("stats/ess.jl")
include("stats/summary.jl")
include("stats/zv.jl")
include("ui.jl") # This is a high level user interface

end
