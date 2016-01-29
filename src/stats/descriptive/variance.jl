### mcvar and mcse stand for Monte Carlo variance and Monte Carlo error respectively

## Monte Carlo variance assuming IID samples

mcvar_iid(v::AbstractArray) = var(v)/length(v)

mcvar_iid(s::VariableNState{Univariate}) = mcvar_iid(s.value)

mcvar_iid(s::VariableNState{Multivariate}, i::Int) = mcvar_iid(s.value[i, :])

mcvar_iid(s::VariableNState{Multivariate}, r::Range=1:s.size) = eltype(s)[mcvar_iid(s, i) for i in r]

## Monte Carlo standard error assuming IID samples

mcse_iid(v::AbstractArray) = sqrt(mcvar_iid(v))

mcse_iid(s::VariableNState{Univariate}) = mcse_iid(s.value)

mcse_iid(s::VariableNState{Multivariate}, i::Int) = mcse_iid(s.value[i, :])

mcse_iid(s::VariableNState{Multivariate}, r::Range=1:s.size) = eltype(s)[mcse_iid(s, i) for i in r]

## Monte Carlo variance using batch means
## Reference:
## Flegal J.M and Jones G.L.
## Batch Means and Spectral Variance Estimators in Markov chain Monte Carlo
## Annals of Statistics, 2010, 38 (2), pp 1034-1070

function mcvar_bm{T}(v::AbstractArray{T}; batchlen::Int=100)
  nbatches = div(length(v), batchlen)
  @assert nbatches > 1 "Choose batch size such that the number of batches is greather than one"
  nbsamples = nbatches*batchlen
  batchmeans = T[mean(v[((j-1)*batchlen+1):(j*batchlen)]) for j = 1:nbatches]
  return batchlen*var(batchmeans)/nbsamples
end

mcvar_bm(s::VariableNState{Univariate}; batchlen::Int=100) = mcvar_bm(s.value, batchlen=batchlen)

mcvar_bm(s::VariableNState{Multivariate}, i::Int; batchlen::Int=100) = mcvar_bm(s.value[i, :], batchlen=batchlen)

mcvar_bm(s::VariableNState{Multivariate}, r::Range=1:s.size; batchlen::Int=100) =
  eltype(s)[mcvar_bm(s, i, batchlen=batchlen) for i in r]

## Monte Carlo standard error using batch means

mcse_bm(v::AbstractArray; batchlen::Int=100) = sqrt(mcvar_bm(v, batchlen=batchlen))

mcse_bm(s::VariableNState{Univariate}; batchlen::Int=100) = mcse_bm(s.value, batchlen=batchlen)

mcse_bm(s::VariableNState{Multivariate}, i::Int; batchlen::Int=100) = mcse_bm(s.value[i, :], batchlen=batchlen)

mcse_bm(s::VariableNState{Multivariate}, r::Range=1:s.size; batchlen::Int=100) =
  eltype(s)[mcse_bm(s, i, batchlen=batchlen) for i in r]

## Initial monotone sequence estimator (IMSE) of Monte Carlo variance
## Reference:
## Geyer C.J.
## Practical Markov Chain Monte Carlo
## Statistical Science, 1992, 7 (4), pp 473-483

function mcvar_imse{T}(v::AbstractVector{T}; maxlag::Int=length(v)-1)
  k = convert(Int, floor((maxlag-1)/2))
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

mcvar_imse(s::VariableNState{Univariate}; maxlag::Int=s.n-1) = mcvar_imse(s.value, maxlag=maxlag)

mcvar_imse(s::VariableNState{Multivariate}, i::Int; maxlag::Int=s.n-1) = mcvar_imse(vec(s.value[i, :]), maxlag=maxlag)

mcvar_imse(s::VariableNState{Multivariate}, r::Range=1:s.size; maxlag::Int=s.n-1) =
  eltype(s)[mcvar_imse(s, i, maxlag=maxlag) for i in r]

## Initial monotone sequence estimator (IMSE) of Monte Carlo standard error

mcse_imse(v::AbstractArray; maxlag::Int=length(v)-1) = sqrt(mcvar_imse(v, maxlag=maxlag))

mcse_imse(s::VariableNState{Univariate}; maxlag::Int=s.n-1) = mcse_imse(s.value, maxlag=maxlag)

mcse_imse(s::VariableNState{Multivariate}, i::Int; maxlag::Int=s.n-1) = mcse_imse(s.value[i, :], maxlag=maxlag)

mcse_imse(s::VariableNState{Multivariate}, r::Range=1:s.size; maxlag::Int=s.n-1) =
  eltype(s)[mcse_imse(s, i, maxlag=maxlag) for i in r]
