### mcvar and mcse stand for Monte Carlo variance and Monte Carlo error respectively

## Monte Carlo variance assuming IID samples

mcvar_iid(v::AbstractArray) = var(v)/length(v)

mcvar_iid(v::AbstractArray, region) = mapslices(mcvar_iid, v, region)

mcvar_iid(s::VariableNState{Univariate}) = mcvar_iid(s.value)

mcvar_iid(s::VariableNState{Multivariate}, i::Integer) = mcvar_iid(s.value[i, :])

mcvar_iid(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size) = eltype(s)[mcvar_iid(s, i) for i in r]

## Monte Carlo standard error assuming IID samples

mcse_iid(v::AbstractArray) = sqrt(mcvar_iid(v))

mcse_iid(v::AbstractArray, region) = mapslices(mcse_iid, v, region)

mcse_iid(s::VariableNState{Univariate}) = mcse_iid(s.value)

mcse_iid(s::VariableNState{Multivariate}, i::Integer) = mcse_iid(s.value[i, :])

mcse_iid(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size) = eltype(s)[mcse_iid(s, i) for i in r]

## Monte Carlo variance using batch means
## Reference:
## Flegal J.M and Jones G.L
## Batch Means and Spectral Variance Estimators in Markov chain Monte Carlo
## Annals of Statistics, 2010, 38 (2), pp 1034-1070

function mcvar_bm{T}(v::AbstractArray{T}; batchlen::Integer=100)
  nbatches = div(length(v), batchlen)
  @assert nbatches > 1 "Choose batch size such that the number of batches is greather than one"
  nbsamples = nbatches*batchlen
  batchmeans = T[mean(v[((j-1)*batchlen+1):(j*batchlen)]) for j = 1:nbatches]
  return batchlen*var(batchmeans)/nbsamples
end

mcvar_bm(v::AbstractArray, region; batchlen::Integer=100) = mapslices(x -> mcvar_bm(x, batchlen=batchlen), v, region)

mcvar_bm(s::VariableNState{Univariate}; batchlen::Integer=100) = mcvar_bm(s.value, batchlen=batchlen)

mcvar_bm(s::VariableNState{Multivariate}, i::Integer; batchlen::Integer=100) = mcvar_bm(s.value[i, :], batchlen=batchlen)

mcvar_bm(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; batchlen::Integer=100) =
  eltype(s)[mcvar_bm(s, i, batchlen=batchlen) for i in r]

## Monte Carlo standard error using batch means

mcse_bm(v::AbstractArray; batchlen::Integer=100) = sqrt(mcvar_bm(v, batchlen=batchlen))

mcse_bm(v::AbstractArray, region; batchlen::Integer=100) = mapslices(x -> mcse_bm(x, batchlen=batchlen), v, region)

mcse_bm(s::VariableNState{Univariate}; batchlen::Integer=100) = mcse_bm(s.value, batchlen=batchlen)

mcse_bm(s::VariableNState{Multivariate}, i::Integer; batchlen::Integer=100) = mcse_bm(s.value[i, :], batchlen=batchlen)

mcse_bm(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; batchlen::Integer=100) =
  eltype(s)[mcse_bm(s, i, batchlen=batchlen) for i in r]

## Initial monotone sequence estimator (IMSE) of Monte Carlo variance
## Reference:
## Geyer C.J
## Practical Markov Chain Monte Carlo
## Statistical Science, 1992, 7 (4), pp 473-483

function mcvar_imse{T}(v::AbstractVector{T}; maxlag::Integer=length(v)-1)
  k = convert(Integer, floor((maxlag-1)/2))
  m = k+1

  # Preallocate memory
  g = Array(T, k+1)

  # Calculate empirical autocovariance
  acv = autocov(v, 0:maxlag)

  # Calculate \hat{G}_{n, m} from the autocovariances, see pp 477 in Geyer
  for j = 0:k
    g[j+1] = acv[2*j+1]+acv[2*j+2]
    if g[j+1] <= 0
      m = j
      break
    end
  end

  # Create the monotone sequence of g
  if m > 1
    for j = 2:m
      if g[j] > g[j-1]
        g[j] = g[j-1]
      end
    end
  end

  # Calculate the initial monotone sequence estimator
  return (-acv[1]+2*sum(g[1:m]))/length(v)
end

mcvar_imse(v::AbstractArray; maxlag::Integer=length(v)-1) = mcvar_imse(vec(v), maxlag=maxlag)

mcvar_imse(v::AbstractArray, region; maxlag::Integer=length(v)-1) = mapslices(x -> mcvar_imse(x, maxlag=maxlag), v, region)

mcvar_imse(v::AbstractArray, region) = mapslices(x -> mcvar_imse(x, maxlag=length(x)-1), v, region)

mcvar_imse(s::VariableNState{Univariate}; maxlag::Integer=s.n-1) = mcvar_imse(s.value, maxlag=maxlag)

mcvar_imse(s::VariableNState{Multivariate}, i::Integer; maxlag::Integer=s.n-1) = mcvar_imse(s.value[i, :], maxlag=maxlag)

mcvar_imse(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; maxlag::Integer=s.n-1) =
  eltype(s)[mcvar_imse(s, i, maxlag=maxlag) for i in r]

## Initial monotone sequence estimator (IMSE) of Monte Carlo standard error

mcse_imse(v::AbstractArray; maxlag::Integer=length(v)-1) = sqrt(mcvar_imse(v, maxlag=maxlag))

mcse_imse(v::AbstractArray, region; maxlag::Integer=length(v)-1) = mapslices(x -> mcse_imse(x, maxlag=maxlag), v, region)

mcse_imse(v::AbstractArray, region) = mapslices(x -> mcse_imse(x, maxlag=length(x)-1), v, region)

mcse_imse(s::VariableNState{Univariate}; maxlag::Integer=s.n-1) = mcse_imse(s.value, maxlag=maxlag)

mcse_imse(s::VariableNState{Multivariate}, i::Integer; maxlag::Integer=s.n-1) = mcse_imse(s.value[i, :], maxlag=maxlag)

mcse_imse(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; maxlag::Integer=s.n-1) =
  eltype(s)[mcse_imse(s, i, maxlag=maxlag) for i in r]

## Initial positive sequence estimator (IPSE) of Monte Carlo variance
## Reference:
## Geyer C.J
## Practical Markov Chain Monte Carlo
## Statistical Science, 1992, 7 (4), pp 473-483

function mcvar_ipse{T}(v::AbstractVector{T}; maxlag::Integer=length(v)-1)
  k = convert(Integer, floor((maxlag-1)/2))

  # Preallocate memory
  g = Array(T, k+1)
  m = k+1

  # Calculate empirical autocovariance
  acv = autocov(v, 0:maxlag)

  # Calculate \hat{G}_{n, m} from the autocovariances, see pp. 477 in Geyer
  for j = 0:k
    g[j+1] = acv[2*j+1]+acv[2*j+2]
    if g[j+1] <= 0
      m = j
      break
    end
  end

  # Calculate the initial positive sequence estimator
  return (-acv[1]+2*sum(g[1:m]))/length(v)
end

mcvar_ipse(v::AbstractArray; maxlag::Integer=length(v)-1) = mcvar_ipse(vec(v), maxlag=maxlag)

mcvar_ipse(v::AbstractArray, region; maxlag::Integer=length(v)-1) = mapslices(x -> mcvar_ipse(x, maxlag=maxlag), v, region)

mcvar_ipse(v::AbstractArray, region) = mapslices(x -> mcvar_ipse(x, maxlag=length(x)-1), v, region)

mcvar_ipse(s::VariableNState{Univariate}; maxlag::Integer=s.n-1) = mcvar_ipse(s.value, maxlag=maxlag)

mcvar_ipse(s::VariableNState{Multivariate}, i::Integer; maxlag::Integer=s.n-1) = mcvar_ipse(s.value[i, :], maxlag=maxlag)

mcvar_ipse(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; maxlag::Integer=s.n-1) =
  eltype(s)[mcvar_ipse(s, i, maxlag=maxlag) for i in r]

## Initial positive sequence estimator (IPSE) of Monte Carlo standard error

mcse_ipse(v::AbstractArray; maxlag::Integer=length(v)-1) = sqrt(mcvar_ipse(v, maxlag=maxlag))

mcse_ipse(v::AbstractArray, region; maxlag::Integer=length(v)-1) = mapslices(x -> mcse_ipse(x, maxlag=maxlag), v, region)

mcse_ipse(v::AbstractArray, region) = mapslices(x -> mcse_ipse(x, maxlag=length(x)-1), v, region)

mcse_ipse(s::VariableNState{Univariate}; maxlag::Integer=s.n-1) = mcse_ipse(s.value, maxlag=maxlag)

mcse_ipse(s::VariableNState{Multivariate}, i::Integer; maxlag::Integer=s.n-1) = mcse_ipse(s.value[i, :], maxlag=maxlag)

mcse_ipse(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; maxlag::Integer=s.n-1) =
  eltype(s)[mcse_ipse(s, i, maxlag=maxlag) for i in r]

## Wrapper function for estimating Monte Carlo variance using various approaches

mcvar_types = (:iid, :bm, :imse, :ipse)
mcvar_functions = Dict{Symbol, Function}(zip(mcvar_types, (mcvar_iid, mcvar_bm, mcvar_imse, mcvar_ipse)))
mcse_functions = Dict{Symbol, Function}(zip(mcvar_types, (mcse_iid, mcse_bm, mcse_imse, mcse_ipse)))

mcvar(v::AbstractArray; vtype::Symbol=:imse, args...) = mcvar_functions[vtype](v; args...)

mcvar(v::AbstractArray, region; vtype::Symbol=:imse, args...) = mcvar_functions[vtype](v, region; args...)

mcvar(s::VariableNState{Univariate}; vtype::Symbol=:imse, args...) = mcvar_functions[vtype](s; args...)

mcvar(s::VariableNState{Multivariate}, i::Integer; vtype::Symbol=:imse, args...) = mcvar_functions[vtype](s, i; args...)

mcvar(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; vtype::Symbol=:imse, args...) =
  mcvar_functions[vtype](s, r; args...)

## Wrapper function for estimating Monte Carlo standard error using various approaches

mcse(v::AbstractArray; vtype::Symbol=:imse, args...) = mcse_functions[vtype](v; args...)

mcse(v::AbstractArray, region; vtype::Symbol=:imse, args...) = mcse_functions[vtype](v, region; args...)

mcse(s::VariableNState{Univariate}; vtype::Symbol=:imse, args...) = mcse_functions[vtype](s; args...)

mcse(s::VariableNState{Multivariate}, i::Integer; vtype::Symbol=:imse, args...) = mcse_functions[vtype](s, i; args...)

mcse(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size; vtype::Symbol=:imse, args...) =
  mcse_functions[vtype](s, r; args...)
