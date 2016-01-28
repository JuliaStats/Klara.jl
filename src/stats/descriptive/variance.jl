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

## Function for estimating Monte Carlo variance using batch means
## Reference:
## Flegal J.M, Jones, G.L.
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

mcse_bm(v::AbstractArray; batchlen::Int=100) = sqrt(mcvar_bm(x; batchlen=batchlen))

mcse_bm(s::VariableNState{Univariate}; batchlen::Int=100) = mcse_bm(s.value, batchlen=batchlen)

mcse_bm(s::VariableNState{Multivariate}, i::Int; batchlen::Int=100) = mcse_bm(s.value[i, :], batchlen=batchlen)

mcse_bm(s::VariableNState{Multivariate}, r::Range=1:s.size; batchlen::Int=100) =
  eltype(s)[mcse_bm(s, i, batchlen=batchlen) for i in r]
