module Lora

using Distributions
using Formatting
using Graphs

import Base:
  ==,
  Dict,
  close,
  convert,
  copy!,
  eltype,
  flush,
  getindex,
  isequal,
  keys,
  mark,
  open,
  read!,
  read,
  reset,
  run,
  show,
  write,
  writemime

import ForwardDiff

import Graphs:
  add_edge!,
  add_vertex!,
  edge_index,
  edges,
  in_degree,
  in_edges,
  in_neighbors,
  is_directed,
  make_edge,
  num_edges,
  num_vertices,
  out_degree,
  out_edges,
  out_neighbors,
  revedge,
  source,
  target,
  topological_sort_by_dfs,
  vertex_index,
  vertices

import ReverseDiffSource

export
  ### Types
  AcceptanceRateMCTune,
  AcceptanceRateMCTuner,
  BasicContMuvParameter,
  BasicContMuvParameterNState,
  BasicContMuvParameterState,
  BasicContParamIOStream,
  BasicContUnvParameter,
  BasicContUnvParameterNState,
  BasicContUnvParameterState,
  BasicMCJob,
  BasicMCRange,
  BasicMavVariableNState,
  BasicMavVariableState,
  BasicMuvVariableNState,
  BasicMuvVariableState,
  BasicUnvVariableNState,
  BasicUnvVariableState,
  BasicVariableIOStream,
  Constant,
  ContMuvMarkovChain,
  ContUnvMarkovChain,
  Data,
  Dependence,
  Deterministic,
  GenericModel,
  GibbsJob,
  HMCSampler,
  Hyperparameter,
  LMCSampler,
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
  MHState,
  MarkovChain,
  MuvMALAState,
  Parameter,
  ParameterIOStream,
  ParameterNState,
  ParameterState,
  Random,
  Sampleability,
  Transformation,
  VanillaMCTune,
  VanillaMCTuner,
  Variable,
  VariableIOStream,
  VariableNState,
  VariableState,

  ### Functions
  add_dimension,
  add_edge!,
  add_vertex!,
  count!,
  edge_index,
  edges,
  erf_rate_score,
  in_degree,
  in_edges,
  in_neighbors,
  indices,
  is_directed,
  is_indexed,
  job2dot,
  likelihood_model,
  logistic,
  logistic_rate_score,
  make_edge,
  model2dot,
  num_edges,
  num_vertices,
  out_degree,
  out_edges,
  out_neighbors,
  output,
  rate!,
  reset!,
  reset_burnin!,
  revedge,
  run,
  sampler_state,
  save!,
  save,
  single_parameter_likelihood_model,
  source,
  target,
  topological_sort_by_dfs,
  tune!,
  tuner_state,
  vertex_index,
  vertex_key,
  vertices

include("format.jl")

include("stats/logistic.jl")

include("autodiff/reverse.jl")
include("autodiff/forward.jl")

include("states/VariableStates.jl")
include("states/ParameterStates.jl")
include("states/VariableNStates.jl")
include("states/ParameterNStates.jl")

include("iostreams/VariableIOStreams.jl")
include("iostreams/ParameterIOStreams.jl")

include("variables/variables.jl")
include("variables/BasicContUnvParameter.jl")
include("variables/BasicContMuvParameter.jl")
include("variables/dependencies.jl")

include("models/GenericModel.jl")
include("models/generators.jl")

include("ranges/ranges.jl")
include("ranges/BasicMCRange.jl")

include("tuners/tuners.jl")
include("tuners/VanillaMCTuner.jl")
include("tuners/AcceptanceRateMCTuner.jl")

include("samplers/samplers.jl")
include("samplers/MH.jl")
include("samplers/MALA.jl")

include("jobs/jobs.jl")
include("jobs/BasicMCJob.jl")
include("jobs/GibbsJob.jl")

include("samplers/iterate/MH.jl")
include("samplers/iterate/MALA.jl")
include("samplers/iterate/iterate.jl")

end
