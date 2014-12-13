module Lora

using Base.LinAlg.BLAS
using Distributions
using StatsBase
using ReverseDiffSource

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
  MCLikelihood,
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
  PlainMCJob,
  TaskMCJob,
  ### functions
  model,
  set_mcjob,
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

include("api/api.jl")
include("api/samples.jl")
include("api/states.jl")
include("api/chains.jl")
include("parsers/expr_funcs.jl")
include("parsers/modelparser.jl")
include("parsers/definitions/DistributionsExtensions.jl")
include("parsers/definitions/AccumulatorDerivRules.jl")
include("parsers/definitions/MCMCDerivRules.jl")
include("parsers/expr_funcs.jl")
include("models/MCLikelihood.jl")
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
include("stats/mean.jl")
include("stats/var.jl")
include("stats/zv.jl")
include("diagnostics/ess.jl")
include("diagnostics/actime.jl")
include("diagnostics/summary.jl")
# Output management: filter, merge, extract, read and write chains, convert btwn chains, arrays and dataframes
# include("output/filter.jl")
# include("output/merge.jl")
include("ui/minimal.jl")

end
