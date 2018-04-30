"""
Basic continuous multivariate parameter state type

# Constructors

## BasicContMuvParameterState{N<:Real}(value::Vector{N}, <optional arguments>)

Construct a basic continuous multivariate parameter state with some ``value``.

###  Optional arguments:

* ``monitor::Vector{Bool}=fill(false, 9)``: 9-element Boolean vector indicating which of the target-related fields are
stored by the state.
* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))``: the diagnostic values of the state.

## BasicContMuvParameterState{N<:Real}(value::Vector{N}, monitor::Vector{Symbol}, <optional arguments>)

Construct a basic continuous multivariate parameter state with some ``value`` and tracked target-related fields specified by
``monitor``.

###  Optional arguments:

* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))``: the diagnostic values of the state.

## BasicContMuvParameterState{N<:Real}(size::Integer, <optional arguments>)

Construct a basic continuous multivariate parameter state with a ``value`` of specified ``size``.

###  Optional arguments:

* ``monitor::Vector{Bool}=fill(false, 9)``: 9-element Boolean vector indicating which of the target-related fields are
stored by the state.
* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{N}=Float64``: the element type of the state value.
* ``diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))``: the diagnostic values of the state.

## BasicContMuvParameterState{N<:Real}(size::Integer, monitor::Vector{Symbol}, <optional arguments>)

Construct a basic continuous multivariate parameter state with a ``value`` of specified ``size`` and tracked target-related
fields specified by ``monitor``.

###  Optional arguments:

* ``diagnostickeys::Vector{Symbol}=Symbol[]``: the diagnostic keys of the state.
* ``::Type{N}=Float64``: the element type of the state value.
* ``diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))``: the diagnostic values of the state.

# Examples

```julia
julia> state = BasicContMuvParameterState(zeros(Float64, 2), [:logtarget])
Klara.BasicContMuvParameterState{Float64}([0.0,0.0],NaN,NaN,NaN,Float64[],Float64[],Float64[],0x0 Array{Float64,2},0x0 Array{Float64,2},0x0 Array{Float64,2},0x0x0 Array{Float64,3},0x0x0 Array{Float64,3},0x0x0 Array{Float64,3},Any[],2,Symbol[])

julia> state.value
2-element Array{Float64,1}:
 0.0
 0.0
```
"""
mutable struct BasicContMuvParameterState{N<:Real} <: ParameterState{Continuous, Multivariate}
  "Vector value of basic continuous multivariate parameter state"
  value::Vector{N}
  "Value of log-likelihood at the state's value"
  loglikelihood::N
  "Value of log-prior at the state's value"
  logprior::N
  "Value of log-target at the state's value"
  logtarget::N
  "Value of gradient of log-likelihood at the state's value"
  gradloglikelihood::Vector{N}
  "Value of gradient of log-prior at the state's value"
  gradlogprior::Vector{N}
  "Value of gradient of log-target at the state's value"
  gradlogtarget::Vector{N}
  "Value of metric tensor of log-likelihood at the state's value"
  tensorloglikelihood::Matrix{N}
  "Value of metric tensor of log-prior at the state's value"
  tensorlogprior::Matrix{N}
  "Value of metric tensor of log-target at the state's value"
  tensorlogtarget::Matrix{N}
  "Value of derivatives of metric tensor of log-likelihood at the state's value"
  dtensorloglikelihood::Array{N, 3}
  "Value of derivatives of metric tensor of log-prior at the state's value"
  dtensorlogprior::Array{N, 3}
  "Value of derivatives of metric tensor of log-target at the state's value"
  dtensorlogtarget::Array{N, 3}
  "Diagnostic values associated with the sampling of the state"
  diagnosticvalues::Vector
  "Integer representing the length of vector value of basic continuous multivariate parameter state"
  size::Integer
  "Diagnostic keys associated with the sampling of the state"
  diagnostickeys::Vector{Symbol}
  diffmethods::Union{DiffMethods, Void}
  diffstate::Union{DiffState, Void}
end

function BasicContMuvParameterState(
  value::Vector{N},
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys)),
) where N<:Real
  v = convert(N, NaN)

  s = length(value)

  l = Array{Integer}(9)
  for i in 1:9
    l[i] = (monitor[i] == false ? zero(Integer) : s)
  end

  diffstate = DiffState()

  if diffopts != nothing
    if diffopts.order == 1
      for (i, field) in ((1, :resultll), (2, :resultlp), (3, :resultlt))
        if diffopts.targets[i]
          setfield!(diffstate, field, DiffResults.GradientResult(value))
        end
      end
    else
      for (i, field) in ((1, :resultll), (2, :resultlp), (3, :resultlt))
        if diffopts.targets[i]
          setfield!(diffstate, field, DiffResults.HessianResult(value))
        end
      end
    end

    if diffopts.mode == :reverse
      for (i, field) in ((4, :cfggll), (5, :cfgglp), (6, :cfgglt))
        if diffopts.targets[i-3]
          setfield!(diffstate, field, ReverseDiff.GradientConfig(value))
        end
      end

      if diffopts.order == 2
        for (i, field) in ((7, :cfgtll), (8, :cfgtlp), (9, :cfgtlt))
          if diffopts.targets[i-6]
            setfield!(diffstate, field, ReverseDiff.HessianConfig(value))
          end
        end
      end
    else
      if diffopts.chunksize == 0
        for (i, field, diffmethod) in ((4, :cfggll, :closurell), (5, :cfgglp, :closurelp), (6, :cfgglt, :closurelt))
          if diffopts.targets[i-3]
            setfield!(diffstate, field, ForwardDiff.GradientConfig(getfield(diffmethods, diffmethod), value))
          end
        end

        if diffopts.order == 2
          for (i, diffmethod, diffresult, diffconfig) in (
            (7, :closurell, :resultll, :cfgtll),
            (8, :closurelp, :resultlp, :cfgtlp),
            (9, :closurelt, :resultlt, :cfgtlt)
          )
            if diffopts.targets[i-6]
              setfield!(
                diffstate,
                diffconfig,
                ForwardDiff.HessianConfig(getfield(diffmethods, diffmethod), getfield(diffstate, diffresult), value)
              )
            end
          end
        end
      else
        for (i, field, diffmethod) in ((4, :cfggll, :closurell), (5, :cfgglp, :closurelp), (6, :cfgglt, :closurelt))
          if diffopts.targets[i-3]
            setfield!(
              diffstate,
              field,
              ForwardDiff.GradientConfig(
                getfield(diffmethods, diffmethod), value, ForwardDiff.Chunk{diffopts.chunksize}()
              )
            )
          end
        end

        if diffopts.order == 2
          for (i, diffmethod, diffresult, diffconfig) in (
            (7, :closurell, :resultll, :cfgtll),
            (8, :closurelp, :resultlp, :cfgtlp),
            (9, :closurelt, :resultlt, :cfgtlt)
          )
            if diffopts.targets[i-6]
              setfield!(
                diffstate,
                diffconfig,
                ForwardDiff.HessianConfig(
                  getfield(diffmethods, diffmethod),
                  getfield(diffstate, diffresult),
                  value,
                  ForwardDiff.Chunk{diffopts.chunksize}()
                )
              )
            end
          end
        end
      end
    end
  end

  BasicContMuvParameterState{N}(
    value,
    v,
    v,
    v,
    Array{N}(l[1]),
    Array{N}(l[2]),
    Array{N}(l[3]),
    Array{N}(l[4], l[4]),
    Array{N}(l[5], l[5]),
    Array{N}(l[6], l[6]),
    Array{N}(l[7], l[7], l[7]),
    Array{N}(l[8], l[8], l[8]),
    Array{N}(l[9], l[9], l[9]),
    diagnosticvalues,
    s,
    diagnostickeys,
    diffmethods,
    diffstate
  )
end

function BasicContMuvParameterState(
  value::Vector{N},
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))
) where N<:Real
  fnames = fieldnames(BasicContMuvParameterState)
  BasicContMuvParameterState(
    value, [fnames[i] in monitor ? true : false for i in 5:13], diagnostickeys, diffmethods, diffopts, diagnosticvalues
  )
end

BasicContMuvParameterState(
  size::Integer,
  monitor::Vector{Bool}=fill(false, 9),
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))
) where {N<:Real} =
  BasicContMuvParameterState(Array{N}(size), monitor, diagnostickeys, diffmethods, diffopts, diagnosticvalues)

BasicContMuvParameterState(
  size::Integer,
  monitor::Vector{Symbol},
  diagnostickeys::Vector{Symbol}=Symbol[],
  ::Type{N}=Float64,
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  diagnosticvalues::Vector=Array{Any}(length(diagnostickeys))
) where {N<:Real} =
  BasicContMuvParameterState(Array{N}(size), monitor, diagnostickeys, diffmethods, diffopts, diagnosticvalues)

value_support{N<:Real}(::Type{BasicContMuvParameterState{N}}) = Continuous
value_support(::BasicContMuvParameterState) = Continuous

variate_form{N<:Real}(::Type{BasicContMuvParameterState{N}}) = Multivariate
variate_form(::BasicContMuvParameterState) = Multivariate

eltype{N<:Real}(::Type{BasicContMuvParameterState{N}}) = N
eltype{N<:Real}(::BasicContMuvParameterState{N}) = N

generate_empty(
  state::BasicContMuvParameterState,
  diffmethods::Union{DiffMethods, Void}=nothing,
  diffopts::Union{DiffOptions, Void}=nothing,
  monitor::Vector{Bool}=[isempty(getfield(state, fieldnames(BasicContMuvParameterState)[i])) ? false : true for i in 5:13]
) =
  BasicContMuvParameterState(state.size, monitor, state.diagnostickeys, eltype(state), diffmethods, diffopts)
