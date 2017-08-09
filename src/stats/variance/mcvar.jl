### mcvar and mcse stand for Monte Carlo variance and Monte Carlo error respectively

## Monte Carlo variance assuming IID samples

mcvar(v::AbstractArray, ::Type{Val{:iid}}) = var(v)/length(v)

mcvar(v::AbstractArray, ::Type{Val{:iid}}, region) = mapslices(x -> mcvar(x, Val{:iid}), v, region)

mcvar(s::VariableNState{Univariate}, ::Type{Val{:iid}}) = mcvar(s.value, Val{:iid})

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:iid}}, i::Integer) = mcvar(s.value[i, :], Val{:iid})

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:iid}}, r::AbstractVector=1:s.size) =
  eltype(s)[mcvar(s, Val{:iid}, i) for i in r]

## Monte Carlo standard error assuming IID samples

mcse(v::AbstractArray, ::Type{Val{:iid}}) = sqrt(mcvar(v, Val{:iid}))

mcse(v::AbstractArray, ::Type{Val{:iid}}, region) = mapslices(x -> mcse(x, Val{:iid}), v, region)

mcse(s::VariableNState{Univariate}, ::Type{Val{:iid}}) = mcse(s.value, Val{:iid})

mcse(s::VariableNState{Multivariate}, ::Type{Val{:iid}}, i::Integer) = mcse(s.value[i, :], Val{:iid})

mcse(s::VariableNState{Multivariate}, ::Type{Val{:iid}}, r::AbstractVector=1:s.size) =
  eltype(s)[mcse(s, Val{:iid}, i) for i in r]

## Monte Carlo variance using batch means
## Reference:
## Flegal J.M and Jones G.L
## Batch Means and Spectral Variance Estimators in Markov chain Monte Carlo
## Annals of Statistics, 2010, 38 (2), pp 1034-1070

function mcvar(v::AbstractVector{T}, ::Type{Val{:bm}}, batchlen::Integer=100) where T
  nbatches = div(length(v), batchlen)
  @assert nbatches > 1 "Choose batch size such that the number of batches is greather than one"
  nbsamples = nbatches*batchlen
  batchmeans = T[mean(v[((j-1)*batchlen+1):(j*batchlen)]) for j = 1:nbatches]
  return batchlen*var(batchmeans)/nbsamples
end

mcvar(v::AbstractArray, ::Type{Val{:bm}}, region, batchlen::Integer=100) =
  mapslices(x -> mcvar(vec(x), Val{:bm}, batchlen), v, region)

mcvar(s::VariableNState{Univariate}, ::Type{Val{:bm}}, batchlen::Integer=100) = mcvar(s.value, Val{:bm}, batchlen)

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:bm}}, i::Integer, batchlen::Integer=100) =
  mcvar(s.value[i, :], Val{:bm}, batchlen)

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:bm}}, r::AbstractVector=1:s.size, batchlen::Integer=100) =
  eltype(s)[mcvar(s, Val{:bm}, i, batchlen) for i in r]

## Monte Carlo standard error using batch means

mcse(v::AbstractVector, ::Type{Val{:bm}}, batchlen::Integer=100) = sqrt(mcvar(v, Val{:bm}, batchlen))

mcse(v::AbstractArray, ::Type{Val{:bm}}, region, batchlen::Integer=100) =
  mapslices(x -> mcse(vec(x), Val{:bm}, batchlen), v, region)

mcse(s::VariableNState{Univariate}, ::Type{Val{:bm}}, batchlen::Integer=100) = mcse(s.value, Val{:bm}, batchlen)

mcse(s::VariableNState{Multivariate}, ::Type{Val{:bm}}, i::Integer, batchlen::Integer=100) =
  mcse(s.value[i, :], Val{:bm}, batchlen)

mcse(s::VariableNState{Multivariate}, ::Type{Val{:bm}}, r::AbstractVector=1:s.size, batchlen::Integer=100) =
  eltype(s)[mcse(s, Val{:bm}, i, batchlen) for i in r]

## Initial monotone sequence estimator (IMSE) of Monte Carlo variance
## Reference:
## Geyer C.J
## Practical Markov Chain Monte Carlo
## Statistical Science, 1992, 7 (4), pp 473-483

function mcvar(v::AbstractVector{T}, ::Type{Val{:imse}}, maxlag::Integer=length(v)-1) where T
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

mcvar(v::AbstractArray, ::Type{Val{:imse}}, region) = mapslices(x -> mcvar(vec(x), Val{:imse}, length(x)-1), v, region)

mcvar(s::VariableNState{Univariate}, ::Type{Val{:imse}}, maxlag::Integer=s.n-1) = mcvar(s.value, Val{:imse}, maxlag)

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:imse}}, i::Integer, maxlag::Integer=s.n-1) =
  mcvar(s.value[i, :], Val{:imse}, maxlag)

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:imse}}, r::AbstractVector=1:s.size, maxlag::Integer=s.n-1) =
  eltype(s)[mcvar(s, Val{:imse}, i, maxlag) for i in r]

## Initial monotone sequence estimator (IMSE) of Monte Carlo standard error

mcse(v::AbstractVector, ::Type{Val{:imse}}, maxlag::Integer=length(v)-1) = sqrt(mcvar(v, Val{:imse}, maxlag))

mcse(v::AbstractArray, ::Type{Val{:imse}}, region) = mapslices(x -> mcse(vec(x), Val{:imse}, length(x)-1), v, region)

mcse(s::VariableNState{Univariate}, ::Type{Val{:imse}}, maxlag::Integer=s.n-1) = mcse(s.value, Val{:imse}, maxlag)

mcse(s::VariableNState{Multivariate}, ::Type{Val{:imse}}, i::Integer, maxlag::Integer=s.n-1) =
  mcse(s.value[i, :], Val{:imse}, maxlag)

mcse(s::VariableNState{Multivariate}, ::Type{Val{:imse}}, r::AbstractVector=1:s.size, maxlag::Integer=s.n-1) =
  eltype(s)[mcse(s, Val{:imse}, i, maxlag) for i in r]

## Initial positive sequence estimator (IPSE) of Monte Carlo variance
## Reference:
## Geyer C.J
## Practical Markov Chain Monte Carlo
## Statistical Science, 1992, 7 (4), pp 473-483

function mcvar(v::AbstractVector{T}, ::Type{Val{:ipse}}, maxlag::Integer=length(v)-1) where T
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

mcvar(v::AbstractArray, ::Type{Val{:ipse}}, region) = mapslices(x -> mcvar(vec(x), Val{:ipse}, length(x)-1), v, region)

mcvar(s::VariableNState{Univariate}, ::Type{Val{:ipse}}, maxlag::Integer=s.n-1) = mcvar(s.value, Val{:ipse}, maxlag)

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:ipse}}, i::Integer, maxlag::Integer=s.n-1) =
  mcvar(s.value[i, :], Val{:ipse}, maxlag)

mcvar(s::VariableNState{Multivariate}, ::Type{Val{:ipse}}, r::AbstractVector=1:s.size, maxlag::Integer=s.n-1) =
  eltype(s)[mcvar(s, Val{:ipse}, i, maxlag) for i in r]

## Initial positive sequence estimator (IPSE) of Monte Carlo standard error

mcse(v::AbstractVector, ::Type{Val{:ipse}}, maxlag::Integer=length(v)-1) = sqrt(mcvar(v, Val{:ipse}, maxlag))

mcse(v::AbstractArray, ::Type{Val{:ipse}}, region) = mapslices(x -> mcse(vec(x), Val{:ipse}, length(x)-1), v, region)

mcse(s::VariableNState{Univariate}, ::Type{Val{:ipse}}, maxlag::Integer=s.n-1) = mcse(s.value, Val{:ipse}, maxlag)

mcse(s::VariableNState{Multivariate}, ::Type{Val{:ipse}}, i::Integer, maxlag::Integer=s.n-1) =
  mcse(s.value[i, :], Val{:ipse}, maxlag)

mcse(s::VariableNState{Multivariate}, ::Type{Val{:ipse}}, r::AbstractVector=1:s.size, maxlag::Integer=s.n-1) =
  eltype(s)[mcse(s, Val{:ipse}, i, maxlag) for i in r]

## Wrapper function for estimating Monte Carlo variance using various approaches

mcvar(v::AbstractArray, vtype::Symbol, args...) = mcvar(v, Val{vtype}, args...)

mcvar(s::VariableNState{Univariate}, vtype::Symbol, args...) = mcvar(s, Val{vtype}, args...)

mcvar(s::VariableNState{Multivariate}, vtype::Symbol, i::Integer, args...) = mcvar(s, Val{vtype}, i, args...)

mcvar(s::VariableNState{Multivariate}, vtype::Symbol, r::AbstractVector=1:s.size, args...) = mcvar(s, Val{vtype}, r, args...)

mcvar(v::AbstractArray, args...) = mcvar(v, Val{:imse}, args...)

mcvar(s::VariableNState{Univariate}, args...) = mcvar(s, Val{:imse}, args...)

mcvar(s::VariableNState{Multivariate}, i::Integer, args...) = mcvar(s, Val{:imse}, i, args...)

mcvar(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size, args...) = mcvar(s, Val{:imse}, r, args...)

## Wrapper function for estimating Monte Carlo standard error using various approaches

mcse(v::AbstractArray, vtype::Symbol, args...) = mcse(v, Val{vtype}, args...)

mcse(s::VariableNState{Univariate}, vtype::Symbol, args...) = mcse(s, Val{vtype}, args...)

mcse(s::VariableNState{Multivariate}, vtype::Symbol, i::Integer, args...) = mcse(s, Val{vtype}, i, args...)

mcse(s::VariableNState{Multivariate}, vtype::Symbol, r::AbstractVector=1:s.size, args...) = mcse(s, Val{vtype}, r, args...)

mcse(v::AbstractArray, args...) = mcse(v, Val{:imse}, args...)

mcse(s::VariableNState{Univariate}, args...) = mcse(s, Val{:imse}, args...)

mcse(s::VariableNState{Multivariate}, i::Integer, args...) = mcse(s, Val{:imse}, i, args...)

mcse(s::VariableNState{Multivariate}, r::AbstractVector=1:s.size, args...) = mcse(s, Val{:imse}, r, args...)
