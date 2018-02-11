__precompile__()

module Klara

using DiffResults
using Distributions
using Formatting
using StatsBase

import Base:
  ==,
  close,
  copy!,
  eltype,
  flush,
  getindex,
  isequal,
  keys,
  mark,
  mean,
  mean!,
  open,
  rand,
  read,
  read!,
  reset,
  run,
  show,
  write

import Distributions:
  @check_args,
  @distr_support,
  failprob,
  logpdf,
  params,
  pdf,
  succprob

import ForwardDiff

import ReverseDiff

export
  ### Types
  AM,
  AMState,
  AMWG,
  AMWGState,
  ARS,
  ARSState,
  AcceptanceRateMCTuner,
  BasicContMuvParameter,
  BasicContMuvParameterNState,
  BasicContMuvParameterState,
  BasicContParamIOStream,
  BasicContUnvParameter,
  BasicContUnvParameterNState,
  BasicContUnvParameterState,
  BasicDiscMuvParameter,
  BasicDiscMuvParameterNState,
  BasicDiscMuvParameterState,
  BasicDiscParamIOStream,
  BasicDiscUnvParameter,
  BasicDiscUnvParameterNState,
  BasicDiscUnvParameterState,
  BasicGibbsJob,
  BasicMCJob,
  BasicMCRange,
  BasicMCTune,
  BasicMavVariableNState,
  BasicMavVariableState,
  BasicMuvVariableNState,
  BasicMuvVariableState,
  BasicUnvVariableNState,
  BasicUnvVariableState,
  BasicVariableIOStream,
  Binary,
  Constant,
  ContMuvMarkovChain,
  ContUnvMarkovChain,
  ContinuousParameter,
  ContinuousParameterNState,
  ContinuousParameterState,
  Data,
  Dependence,
  DependenceVector,
  Deterministic,
  DiffMethods,
  DiffOptions,
  DiffState,
  DiscMuvMarkovChain,
  DiscUnvMarkovChain,
  DiscreteParameter,
  DiscreteParameterNState,
  DiscreteParameterState,
  DualAveragingMCTuner,
  GenericModel,
  GibbsJob,
  HMC,
  HMCSampler,
  HMCSamplerState,
  HMCState,
  Hyperparameter,
  IntegerVector,
  LMCSampler,
  LMCSamplerState,
  MALA,
  MALAState,
  MCJob,
  MCRange,
  MCSampler,
  MCSamplerState,
  MCTuner,
  MCTunerState,
  MH,
  MHSampler,
  MHSamplerState,
  MHState,
  MarkovChain,
  MatrixvariateParameter,
  MatrixvariateParameterNState,
  MatrixvariateParameterState,
  MultivariateGMM,
  MultivariateParameter,
  MultivariateParameterNState,
  MultivariateParameterState,
  MuvAMState,
  MuvAMWG,
  MuvAMWGState,
  MuvHMCState,
  MuvMALAState,
  MuvNUTSState,
  MuvRobertsRosenthalMCTune,
  MuvSMMALAState,
  NUTS,
  NUTSState,
  Parameter,
  ParameterIOStream,
  ParameterNState,
  ParameterState,
  ParameterStateVector,
  ParameterVector,
  RAM,
  RAMState,
  Random,
  RealMatrix,
  RealNormal,
  RealVector,
  RobertsRosenthalMCTune,
  RobertsRosenthalMCTuner,
  SMMALA,
  SMMALAState,
  Sampleability,
  SliceSampler,
  SliceSamplerState,
  Transformation,
  UnivariateParameter,
  UnivariateParameterNState,
  UnivariateParameterState,
  UnvAMState,
  UnvAMWG,
  UnvAMWGState,
  UnvHMCState,
  UnvMALAState,
  UnvNUTSState,
  UnvRobertsRosenthalMCTune,
  UnvSMMALAState,
  VanillaMCTuner,
  Variable,
  VariableIOStream,
  VariableNState,
  VariableState,
  VariableStateVector,
  VariableVector,

  ### Functions
  ==,
  acceptance,
  add_dimension,
  add_edge!,
  add_vertex!,
  close,
  copy!,
  count!,
  covariance!,
  dataset,
  datasets,
  diagnostics,
  edge_index,
  edges,
  eltype,
  erf_rate_score,
  ess,
  examples,
  failprob,
  flush,
  getindex,
  iact,
  in_degree,
  in_edges,
  in_neighbors,
  indexes,
  is_directed,
  is_indexed,
  isequal,
  iterate,
  iterate!,
  job2dot,
  keys,
  likelihood_model,
  logistic,
  logistic_rate_score,
  lognormalise,
  logpdf,
  lzv,
  make_edge,
  mark,
  mean,
  mean!,
  mcse,
  mcvar,
  model2dot,
  multivecs,
  normalise,
  num_edges,
  num_vertices,
  open,
  out_degree,
  out_edges,
  out_neighbors,
  output,
  params,
  pdf,
  qzv,
  rate!,
  read,
  read!,
  recursive_mean,
  recursive_mean!,
  reset,
  reset!,
  reset_burnin!,
  revedge,
  run,
  sampler_state,
  save!,
  save,
  set_gmm,
  set_gmm!,
  set_normal,
  set_normal!,
  setpdf!,
  setprior!,
  show,
  softabs,
  sort_by_index,
  source,
  succprob,
  target,
  tune!,
  tuner_state,
  vertex_index,
  vertex_key,
  vertices,
  write

include("base.jl")

include("format.jl")

include("data.jl")

include("stats/logistic.jl")

include("distributions/Binary.jl")
include("distributions/TruncatedNormal.jl")

include("autodiff/autodiff.jl")
include("autodiff/reverse.jl")
include("autodiff/forward.jl")

include("states/VariableStates.jl")
include("states/ParameterStates/ParameterStates.jl")
include("states/ParameterStates/BasicDiscUnvParameterState.jl")
include("states/ParameterStates/BasicDiscMuvParameterState.jl")
include("states/ParameterStates/BasicContUnvParameterState.jl")
include("states/ParameterStates/BasicContMuvParameterState.jl")

include("nstates/VariableNStates.jl")
include("nstates/ParameterNStates/ParameterNStates.jl")
include("nstates/ParameterNStates/BasicDiscUnvParameterNState.jl")
include("nstates/ParameterNStates/BasicDiscMuvParameterNState.jl")
include("nstates/ParameterNStates/BasicContUnvParameterNState.jl")
include("nstates/ParameterNStates/BasicContMuvParameterNState.jl")

include("iostreams/VariableIOStreams.jl")
include("iostreams/ParameterIOStreams/ParameterIOStreams.jl")
include("iostreams/ParameterIOStreams/BasicDiscParamIOStream.jl")
include("iostreams/ParameterIOStreams/BasicContParamIOStream.jl")

include("variables/variables.jl")
include("variables/parameters/parameters.jl")
include("variables/parameters/BasicDiscUnvParameter.jl")
include("variables/parameters/BasicDiscMuvParameter.jl")
include("variables/parameters/BasicContUnvParameter.jl")
include("variables/parameters/BasicContMuvParameter.jl")
include("variables/dependencies.jl")

include("models/GenericModel.jl")
include("models/generators.jl")

include("ranges/ranges.jl")
include("ranges/BasicMCRange.jl")

include("tuners/tuners.jl")
include("tuners/VanillaMCTuner.jl")
include("tuners/AcceptanceRateMCTuner.jl")
include("tuners/RobertsRosenthalMCTuner.jl")
include("tuners/DualAveragingMCTuner.jl")

include("samplers/samplers.jl")
include("samplers/ARS.jl")
include("samplers/SliceSampler.jl")
include("samplers/MH.jl")
include("samplers/AM.jl")
include("samplers/RAM.jl")
include("samplers/AMWG.jl")
include("samplers/HMC.jl")
include("samplers/NUTS.jl")
include("samplers/MALA.jl")
include("samplers/SMMALA.jl")

include("jobs/jobs.jl")
include("jobs/BasicMCJob.jl")
include("jobs/BasicGibbsJob.jl")

include("samplers/iterate/ARS.jl")
include("samplers/iterate/SliceSampler.jl")
include("samplers/iterate/MH.jl")
include("samplers/iterate/AM.jl")
include("samplers/iterate/RAM.jl")
include("samplers/iterate/AMWG.jl")
include("samplers/iterate/HMC.jl")
include("samplers/iterate/NUTS.jl")
include("samplers/iterate/MALA.jl")
include("samplers/iterate/SMMALA.jl")

include("stats/acceptance.jl")

include("stats/mean.jl")

include("stats/variance/mcvar.jl")
include("stats/variance/zv.jl")

include("stats/covariance.jl")

include("stats/convergence/ess.jl")
include("stats/convergence/iact.jl")

include("stats/metrics.jl")

end
