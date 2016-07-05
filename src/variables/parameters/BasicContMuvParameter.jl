### BasicContMuvParameter

type BasicContMuvParameter <: Parameter{Continuous, Multivariate}
  key::Symbol
  index::Integer
  pdf::Union{ContinuousMultivariateDistribution, Void}
  prior::Union{ContinuousMultivariateDistribution, Void}
  setpdf::Union{Function, Void}
  setprior::Union{Function, Void}
  loglikelihood!::Union{Function, Void}
  logprior!::Union{Function, Void}
  logtarget!::Union{Function, Void}
  gradloglikelihood!::Union{Function, Void}
  gradlogprior!::Union{Function, Void}
  gradlogtarget!::Union{Function, Void}
  tensorloglikelihood!::Union{Function, Void}
  tensorlogprior!::Union{Function, Void}
  tensorlogtarget!::Union{Function, Void}
  dtensorloglikelihood!::Union{Function, Void}
  dtensorlogprior!::Union{Function, Void}
  dtensorlogtarget!::Union{Function, Void}
  uptogradlogtarget!::Union{Function, Void}
  uptotensorlogtarget!::Union{Function, Void}
  uptodtensorlogtarget!::Union{Function, Void}
  states::VariableStateVector

  function BasicContMuvParameter(
    key::Symbol,
    index::Integer,
    pdf::Union{ContinuousMultivariateDistribution, Void},
    prior::Union{ContinuousMultivariateDistribution, Void},
    setpdf::Union{Function, Void},
    setprior::Union{Function, Void},
    ll::Union{Function, Void},
    lp::Union{Function, Void},
    lt::Union{Function, Void},
    gll::Union{Function, Void},
    glp::Union{Function, Void},
    glt::Union{Function, Void},
    tll::Union{Function, Void},
    tlp::Union{Function, Void},
    tlt::Union{Function, Void},
    dtll::Union{Function, Void},
    dtlp::Union{Function, Void},
    dtlt::Union{Function, Void},
    uptoglt::Union{Function, Void},
    uptotlt::Union{Function, Void},
    uptodtlt::Union{Function, Void},
    states::VariableStateVector
  )
    args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)
    fnames = fieldnames(BasicContMuvParameter)[5:21]

    # Check that all generic functions have correct signature
    for i in 1:17
      if isa(args[i], Function) &&
        isgeneric(args[i]) &&
        !method_exists(args[i], (BasicContMuvParameterState, VariableStateVector))
        error("$(fnames[i]) has wrong signature")
      end
    end

    new(
      key,
      index,
      pdf,
      prior,
      setpdf,
      setprior,
      ll,
      lp,
      lt,
      gll,
      glp,
      glt,
      tll,
      tlp,
      tlt,
      dtll,
      dtlp,
      dtlt,
      uptoglt,
      uptotlt,
      uptodtlt,
      states
    )
  end
end

function BasicContMuvParameter!(
  parameter::BasicContMuvParameter,
  setpdf::Union{Function, Void},
  setprior::Union{Function, Void},
  ll::Union{Function, Void},
  lp::Union{Function, Void},
  lt::Union{Function, Void},
  gll::Union{Function, Void},
  glp::Union{Function, Void},
  glt::Union{Function, Void},
  tll::Union{Function, Void},
  tlp::Union{Function, Void},
  tlt::Union{Function, Void},
  dtll::Union{Function, Void},
  dtlp::Union{Function, Void},
  dtlt::Union{Function, Void},
  uptoglt::Union{Function, Void},
  uptotlt::Union{Function, Void},
  uptodtlt::Union{Function, Void}
)
  args = (setpdf, setprior, ll, lp, lt, gll, glp, glt, tll, tlp, tlt, dtll, dtlp, dtlt, uptoglt, uptotlt, uptodtlt)

  # Define setpdf (i = 1) and setprior (i = 2)
  for (i, setter, distribution) in ((1, :setpdf, :pdf), (2, :setprior, :prior))
    setfield!(
      parameter,
      setter,
      if isa(args[i], Function)
        eval(codegen_setfield(parameter, distribution, args[i]))
      else
        nothing
      end
    )
  end

  # Define loglikelihood! (i = 3) and gradloglikelihood! (i = 6)
  # plfield stands for parameter likelihood-related field respectively
  for (i, plfield) in ((3, :loglikelihood!), (6, :gradloglikelihood!))
    setfield!(
      parameter,
      plfield,
      if isa(args[i], Function)
        eval(codegen_closure(parameter, args[i]))
      else
        nothing
      end
    )
  end

  # Define logprior! (i = 4) and gradlogprior! (i = 7)
  # ppfield and spfield stand for parameter prior-related field and state prior-related field repsectively
  for (i , ppfield, spfield, f) in ((4, :logprior!, :logprior, logpdf), (7, :gradlogprior!, :gradlogprior, gradlogpdf))
    setfield!(
    parameter,
      ppfield,
      if isa(args[i], Function)
        eval(codegen_closure(parameter, args[i]))
      else
        if (
            isa(parameter.prior, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(parameter.prior), Vector{eltype(parameter.prior)}))
          ) ||
          isa(args[2], Function)
          eval(codegen_target_closure_via_distribution(parameter, :prior, f, spfield))
        else
          nothing
        end
      end
    )
  end

  # Define logtarget! (i = 5) and gradlogtarget! (i = 8)
  # ptfield, plfield and ppfield stand for parameter target, likelihood and prior-related field respectively
  # stfield, slfield and spfield stand for state target, likelihood and prior-related field respectively
  for (i , ptfield, plfield, ppfield, stfield, slfield, spfield, f) in (
    (5, :logtarget!, :loglikelihood!, :logprior!, :logtarget, :loglikelihood, :logprior, logpdf),
    (8, :gradlogtarget!, :gradloglikelihood!, :gradlogprior!, :gradlogtarget, :gradloglikelihood, :gradlogprior, gradlogpdf)
  )
    setfield!(
      parameter,
      ptfield,
      if isa(args[i], Function)
        eval(codegen_closure(parameter, args[i]))
      else
        if isa(args[i-2], Function) && isa(getfield(parameter, ppfield), Function)
          eval(codegen_sumtarget_closure(parameter, plfield, ppfield, stfield, slfield, spfield))
        elseif (
            isa(parameter.pdf, ContinuousMultivariateDistribution) &&
            method_exists(f, (typeof(parameter.pdf), Vector{eltype(parameter.pdf)}))
          ) ||
          isa(args[1], Function)
          eval(codegen_target_closure_via_distribution(parameter, :pdf, f, stfield))
        else
          nothing
        end
      end
    )
  end

  # Define tensorloglikelihood! (i = 9) and dtensorloglikelihood! (i = 12)
  # plfield stands for parameter likelihood-related field respectively
  for (i, plfield) in ((9, :tensorloglikelihood!), (12, :dtensorloglikelihood!))
    setfield!(
      parameter,
      plfield,
      if isa(args[i], Function)
        eval(codegen_closure(parameter, args[i]))
      else
        nothing
      end
    )
  end

  # Define tensorlogprior! (i = 10) and dtensorlogprior! (i = 13)
  # ppfield stands for parameter prior-related field respectively
  for (i, ppfield) in ((10, :tensorlogprior!), (13, :dtensorlogprior!))
    setfield!(
      parameter,
      ppfield,
      if isa(args[i], Function)
        eval(codegen_closure(parameter, args[i]))
      else
        nothing
      end
    )
  end

  # Define tensorlogtarget! (i = 11) and dtensorlogtarget! (i = 14)
  for (i , ptfield, plfield, ppfield, stfield, slfield, spfield) in (
    (
      11,
      :tensorlogtarget!, :tensorloglikelihood!, :tensorlogprior!,
      :tensorlogtarget, :tensorloglikelihood, :tensorlogprior
    ),
    (
      14,
      :dtensorlogtarget!, :dtensorloglikelihood!, :dtensorlogprior!,
      :dtensorlogtarget, :dtensorloglikelihood, :dtensorlogprior
    )
  )
    setfield!(
      parameter,
      ptfield,
      if isa(args[i], Function)
        eval(codegen_closure(parameter, args[i]))
      else
        if isa(args[i-2], Function) && isa(args[i-1], Function)
          eval(codegen_sumtarget_closure(parameter, plfield, ppfield, stfield, slfield, spfield))
        else
          nothing
        end
      end
    )
  end

  # Define uptogradlogtarget!
  setfield!(
    parameter,
    :uptogradlogtarget!,
    if isa(args[15], Function)
      eval(codegen_closure(parameter, args[15]))
    else
      if isa(parameter.logtarget!, Function) && isa(parameter.gradlogtarget!, Function)
        eval(codegen_uptotarget_closures(parameter, [:logtarget!, :gradlogtarget!]))
      else
        nothing
      end
    end
  )

  # Define uptotensorlogtarget!
  setfield!(
    parameter,
    :uptotensorlogtarget!,
    if isa(args[16], Function)
      eval(codegen_closure(parameter, args[16]))
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function)
        eval(codegen_uptotarget_closures(parameter, [:logtarget!, :gradlogtarget!, :tensorlogtarget!]))
      else
        nothing
      end
    end
  )

  # Define uptodtensorlogtarget!
  setfield!(
    parameter,
    :uptodtensorlogtarget!,
    if isa(args[17], Function)
      eval(codegen_closure(parameter, args[17]))
    else
      if isa(parameter.logtarget!, Function) &&
        isa(parameter.gradlogtarget!, Function) &&
        isa(parameter.tensorlogtarget!, Function) &&
        isa(parameter.dtensorlogtarget!, Function)
        eval(codegen_uptotarget_closures(parameter, [:logtarget!, :gradlogtarget!, :tensorlogtarget!, :dtensorlogtarget!]))
      else
        nothing
      end
    end
  )
end

BasicContMuvParameter(key::Symbol, index::Integer=0; signature::Symbol=:high, args...) =
  BasicContMuvParameter(key, Val{signature}, index; args...)

function BasicContMuvParameter(
  key::Symbol,
  ::Type{Val{:low}},
  index::Integer=0;
  pdf::Union{ContinuousMultivariateDistribution, Void}=nothing,
  prior::Union{ContinuousMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Void}=nothing,
  logprior::Union{Function, Void}=nothing,
  logtarget::Union{Function, Void}=nothing,
  gradloglikelihood::Union{Function, Void}=nothing,
  gradlogprior::Union{Function, Void}=nothing,
  gradlogtarget::Union{Function, Void}=nothing,
  tensorloglikelihood::Union{Function, Void}=nothing,
  tensorlogprior::Union{Function, Void}=nothing,
  tensorlogtarget::Union{Function, Void}=nothing,
  dtensorloglikelihood::Union{Function, Void}=nothing,
  dtensorlogprior::Union{Function, Void}=nothing,
  dtensorlogtarget::Union{Function, Void}=nothing,
  uptogradlogtarget::Union{Function, Void}=nothing,
  uptotensorlogtarget::Union{Function, Void}=nothing,
  uptodtensorlogtarget::Union{Function, Void}=nothing,
  states::VariableStateVector=VariableState[]
)
  parameter = BasicContMuvParameter(key, index, pdf, prior, fill(nothing, 17)..., states)

  BasicContMuvParameter!(
    parameter,
    setpdf,
    setprior,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorloglikelihood,
    tensorlogprior,
    tensorlogtarget,
    dtensorloglikelihood,
    dtensorlogprior,
    dtensorlogtarget,
    uptogradlogtarget,
    uptotensorlogtarget,
    uptodtensorlogtarget
  )

  parameter
end

function BasicContMuvParameter(
  key::Symbol,
  ::Type{Val{:high}},
  index::Integer=0;
  pdf::Union{ContinuousMultivariateDistribution, Void}=nothing,
  prior::Union{ContinuousMultivariateDistribution, Void}=nothing,
  setpdf::Union{Function, Void}=nothing,
  setprior::Union{Function, Void}=nothing,
  loglikelihood::Union{Function, Expr, Void}=nothing,
  logprior::Union{Function, Expr, Void}=nothing,
  logtarget::Union{Function, Expr, Void}=nothing,
  gradloglikelihood::Union{Function, Void}=nothing,
  gradlogprior::Union{Function, Void}=nothing,
  gradlogtarget::Union{Function, Void}=nothing,
  tensorloglikelihood::Union{Function, Void}=nothing,
  tensorlogprior::Union{Function, Void}=nothing,
  tensorlogtarget::Union{Function, Void}=nothing,
  dtensorloglikelihood::Union{Function, Void}=nothing,
  dtensorlogprior::Union{Function, Void}=nothing,
  dtensorlogtarget::Union{Function, Void}=nothing,
  uptogradlogtarget::Union{Function, Void}=nothing,
  uptotensorlogtarget::Union{Function, Void}=nothing,
  uptodtensorlogtarget::Union{Function, Void}=nothing,
  states::VariableStateVector=VariableState[],
  nkeys::Integer=0,
  vfarg::Bool=false,
  autodiff::Symbol=:none,
  order::Integer=1,
  chunksize::Integer=0,
  init::Vector=fill(Any[], 3)
)
  inargs = (
    setpdf,
    setprior,
    loglikelihood,
    logprior,
    logtarget,
    gradloglikelihood,
    gradlogprior,
    gradlogtarget,
    tensorloglikelihood,
    tensorlogprior,
    tensorlogtarget,
    dtensorloglikelihood,
    dtensorlogprior,
    dtensorlogtarget,
    uptogradlogtarget,
    uptotensorlogtarget,
    uptodtensorlogtarget
  )

  fnames = Array(Any, 17)
  fnames[1:2] = fill(Symbol[], 2)
  fnames[3:14] = [Symbol[f] for f in fieldnames(BasicContMuvParameterState)[2:13]]
  for i in 1:3
    fnames[14+i] = Symbol[fnames[j][1] for j in 5:3:(5+i*3)]
  end

  for i in 3:5
    if isa(inargs[i], Expr) && autodiff != :reverse
      error("The only case $(fnames[i][1]) can be an expression is when used in conjunction with reverse mode autodiff")
    end
  end

  if nkeys > 0
    if (autodiff == :forward || autodiff == :reverse) && vfarg
      error("In the case of autodiff, if nkeys is not 0, then vfarg must be false")
    end
  elseif nkeys < 0
    "nkeys must be non-negative, got $nkeys"
  end

  if !in(autodiff, (:none, :forward, :reverse))
    error("autodiff must be :nore or :forward or :reverse, got $autodiff")
  end

  if order < 0 || order > 2
    error("Derivative order must be 0, 1 or 2, got $order")
  elseif autodiff != :reverse && order == 0
    error("Zero order can be used only with reverse mode autodiff")
  end

  @assert chunksize >= 0 "chunksize must be non-negative, got $chunksize"

  initarg = Array(Any, 3)
  initlen = length(init)

  if initlen == 1 || initlen == 2
    if autodiff != :reverse
      @assert all(isempty, init) "init option is used only for reverse mode autodiff"
    end

    for i in 1:3
      initarg[i] = (inargs[i+2] != nothing) ? init : Any[]
    end
  elseif initlen == 3
    if autodiff != :reverse
      @assert all(isempty, init) "init option is used only for reverse mode autodiff"
    end

    initarg = init
  else
    error("init must be a vector of length 1, 2 or 3, got vector of length $initlen")
  end

  parameter = BasicContMuvParameter(key, index, pdf, prior, fill(nothing, 17)..., states)

  outargs = Union{Function, Void}[nothing for i in 1:17]

  for i in 1:17
    if isa(inargs[i], Function)
      outargs[i] = eval(
        codegen_lowlevel_variable_method(inargs[i], :BasicContMuvParameterState, true, fnames[i], nkeys, vfarg)
      )
    end
  end

  if autodiff == :forward
    fadclosure = Array(Union{Function, Void}, 3)
    for i in 3:5
      fadclosure[i-2] =
        if isa(inargs[i], Function)
          nkeys == 0 ? inargs[i] : eval(codegen_internal_autodiff_closure(parameter, inargs[i], nkeys))
        else
          nothing
        end
    end

    for i in 6:8
      if !isa(inargs[i], Function) && isa(inargs[i-3], Function)
        outargs[i] = eval(codegen_lowlevel_variable_method(
          eval(codegen_forward_autodiff_function(Val{:gradient}, fadclosure[i-5], chunksize)),
          :BasicContMuvParameterState,
          true,
          fnames[i],
          0
        ))
      end
    end

    if !isa(inargs[15], Function) && isa(inargs[5], Function)
      outargs[15] = eval(codegen_lowlevel_variable_method(
        eval(codegen_forward_autodiff_uptofunction(Val{:gradient}, fadclosure[3], chunksize)),
        :BasicContMuvParameterState,
        true,
        fnames[15],
        0
      ))
    end

    if order >= 2
      for i in 9:11
        if !isa(inargs[i], Function) && isa(inargs[i-6], Function)
          outargs[i] = eval(codegen_lowlevel_variable_method(
            eval(codegen_forward_autodiff_target(:hessian, fadclosure[i-8], chunksize)),
            :BasicContMuvParameterState,
            true,
            fnames[i],
            0
          ))
        end
      end

      if !isa(inargs[16], Function) && isa(inargs[5], Function)
        outargs[16] = eval(codegen_lowlevel_variable_method(
          eval(codegen_forward_autodiff_uptotarget(:hessian, fadclosure[3], chunksize)),
          :BasicContMuvParameterState,
          true,
          fnames[16],
          0
        ))
      end
    end
  elseif autodiff == :reverse
    local f::Function

    for i in 3:5
      if isa(inargs[i], Expr)
        if nkeys == 0
          f = eval(codegen_reverse_autodiff_function(inargs[i], :Vector, initarg[i-2][1], 0, false))
        else
          f = eval(codegen_reverse_autodiff_function(inargs[i], :Vector, initarg[i-2], 0, false))
          f = eval(codegen_internal_autodiff_closure(parameter, f, nkeys))
        end

        outargs[i] = eval(codegen_lowlevel_variable_method(f, :BasicContMuvParameterState, true, fnames[i], 0))
      end
    end

    for i in 6:8
      if !isa(inargs[i], Function)
        if isa(inargs[i-3], Function)
          if nkeys == 0
            f = ReverseDiffSource.rdiff(inargs[i-3], (initarg[i-5][1][2],), order=1, allorders=false)
          else
            f = ReverseDiffSource.rdiff(
              inargs[i-3], (initarg[i-5][1][2], initarg[i-5][2][2]), ignore=[initarg[i-5][2][1]], order=1, allorders=false
            )
            f = eval(codegen_internal_autodiff_closure(parameter, f, nkeys))
          end

          outargs[i] = eval(codegen_lowlevel_variable_method(f, :BasicContMuvParameterState, true, fnames[i], 0))
        elseif isa(inargs[i-3], Expr)
          if nkeys == 0
            f = eval(codegen_reverse_autodiff_function(inargs[i-3], :Vector, initarg[i-5][1], 1, false))
          else
            f = eval(codegen_reverse_autodiff_function(inargs[i-3], :Vector, initarg[i-5], 1, false))
            f = eval(codegen_internal_autodiff_closure(parameter, f, nkeys))
          end

          outargs[i] = eval(codegen_lowlevel_variable_method(f, :BasicContMuvParameterState, true, fnames[i], 0))
        end
      end
    end

    if !isa(inargs[15], Function)
      if isa(inargs[5], Function)
        if nkeys == 0
          f = ReverseDiffSource.rdiff(inargs[5], (initarg[3][1][2],), order=1, allorders=true)
        else
          f = ReverseDiffSource.rdiff(
            inargs[5], (initarg[3][1][2], initarg[3][2][2]), ignore=[initarg[3][2][1]], order=1, allorders=true
          )
          f = eval(codegen_internal_autodiff_closure(parameter, f, nkeys))
        end

        outargs[15] = eval(codegen_lowlevel_variable_method(f, :BasicContMuvParameterState, true, fnames[15], 0))
      elseif isa(inargs[5], Expr)
        if nkeys == 0
          f = eval(codegen_reverse_autodiff_function(inargs[5], :Vector, initarg[3][1], 1, true))
        else
          f = eval(codegen_reverse_autodiff_function(inargs[5], :Vector, initarg[3], 1, true))
          f = eval(codegen_internal_autodiff_closure(parameter, f, nkeys))
        end

        outargs[15] = eval(codegen_lowlevel_variable_method(f, :BasicContMuvParameterState, true, fnames[15], 0))
      end
    end

    if order >= 2
      for i in 9:11
        if !isa(inargs[i], Function)
          if isa(inargs[i-6], Function) || isa(inargs[i-6], Expr)
            if nkeys == 0
              f = eval(codegen_reverse_autodiff_target(:hessian, inargs[i-6], :Vector, initarg[i-8][1]))
            else
              f = eval(codegen_reverse_autodiff_target(:hessian, inargs[i-6], :Vector, initarg[i-8]))
              f = eval(codegen_internal_autodiff_closure(parameter, f, nkeys))
            end

            outargs[i] = eval(codegen_lowlevel_variable_method(f, :BasicContMuvParameterState, true, fnames[i], 0))
          end
        end
      end

      if !isa(inargs[16], Function)
        if isa(inargs[5], Function) || isa(inargs[5], Expr)
          if nkeys == 0
            f = eval(codegen_reverse_autodiff_uptotarget(:hessian, inargs[5], :Vector, initarg[3][1]))
          else
            f = eval(codegen_reverse_autodiff_uptotarget(:hessian, inargs[5], :Vector, initarg[3]))
            f = eval(codegen_internal_autodiff_closure(parameter, f, nkeys))
          end

          outargs[16] = eval(codegen_lowlevel_variable_method(f, :BasicContMuvParameterState, true, fnames[16], 0))
        end
      end
    end
  end

  BasicContMuvParameter!(parameter, outargs...)

  parameter
end

function codegen_internal_autodiff_closure(parameter::BasicContMuvParameter, f::Function, nkeys::Integer)
  fstatesarg = [Expr(:ref, :Any, [:($(parameter).states[$i].value) for i in 1:nkeys]...)]

  @gensym internal_forward_autodiff_closure

  quote
    function $internal_forward_autodiff_closure(_x::Vector)
      $(f)(_x, $(fstatesarg...))
    end
  end
end

value_support(::Type{BasicContMuvParameter}) = Continuous
value_support(::BasicContMuvParameter) = Continuous

variate_form(::Type{BasicContMuvParameter}) = Multivariate
variate_form(::BasicContMuvParameter) = Multivariate

default_state_type(::BasicContMuvParameter) = BasicContMuvParameterState

default_state{N<:Real}(variable::BasicContMuvParameter, value::Vector{N}, outopts::Dict) =
  BasicContMuvParameterState(
    value,
    [getfield(variable, fieldnames(BasicContMuvParameter)[i]) == nothing ? false : true for i in 10:18],
    (haskey(outopts, :diagnostics) && in(:accept, outopts[:diagnostics])) ? [:accept] : Symbol[]
  )

Base.show(io::IO, ::Type{BasicContMuvParameter}) = print(io, "BasicContMuvParameter")
Base.writemime(io::IO, ::MIME"text/plain", t::Type{BasicContMuvParameter}) = show(io, t)
