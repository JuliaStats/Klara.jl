# Autocorrelation for range
function acf{T<:Real}(x::AbstractVector{T}, lags::Ranges=0:min(length(x)-1, 10log10(length(x)));
  correlation::Bool=true, demean::Bool=true)
  lx, llags = length(x), length(lags)
  if max(lags) > lx; error("Autocovariance distance must be less than sample size"); end

  xs = Array(T, lx)
  demean ? (mx = mean(x); for i = 1:lx; xs[i] = x[i]-mx; end) : (for i = 1:lx; xs[i] = x[i]; end)

  autocov_sumterm = Array(T, llags)
  for i in 1:llags
    autocov_sumterm[i] = dot(xs[1:end-lags[i]], xs[lags[i]+1:end])
  end
  autocov_sumterm

  correlation ? (return autocov_sumterm/((lx-1)*var(x))) : (return autocov_sumterm/lx)
end

# Autocorrelation at a specific lag
acf{T<:Real}(x::AbstractVector{T}, lags::Real; correlation::Bool=true, demean::Bool=true) =
  acf(x, lags:lags, correlation=correlation, demean=demean)[1]

# Cross-correlation for range
function acf{T<:Real}(x::AbstractVector{T}, y::AbstractVector{T}, lags::Ranges=0:min(length(x)-1, 10log10(length(x)));
  correlation::Bool=true, demean::Bool=true)
  lx, ly, llags = length(x), length(y), length(lags)
  if lx != ly error("Input vectors must have same length") end
  if max(lags) > lx; error("Cross-covariance distance must be less than sample size"); end

  xs, ys = Array(T, lx), Array(T, ly)
  if demean
    mx, my = mean(x), mean(y); for i = 1:lx; xs[i], ys[i] = x[i]-mx, y[i]-my; end
  else
    for i = 1:lx; xs[i], ys[i] = x[i], y[i]; end
  end

  crosscov_sumterm = Array(T, llags)
  for i in 1:llags
    crosscov_sumterm[i] = dot(xs[1:end-lags[i]], ys[lags[i]+1:end])
  end
  crosscov_sumterm

  correlation ? (return crosscov_sumterm/((lx-1)*std(x)*std(y))) : (return crosscov_sumterm/lx)
end

# Cross-correlation at a specific lag
acf{T<:Real}(x::AbstractVector{T}, y::AbstractVector{T}, lags::Real; correlation::Bool=true, demean::Bool=true) =
  acf(x, y, lags:lags, correlation=correlation, demean=demean)[1]

# Cross-correlation between all pairs of columns of a matrix for range
function acfall{T<:Real}(x::AbstractMatrix{T}, lags::Ranges=0:min(size(x, 1)-1, 10log10(size(x, 1)));
  correlation::Bool=true, demean::Bool=true)
  ncols = size(x, 2)

  crosscorr = Array(T, length(lags), ncols, ncols)
  for i = 1:ncols
    for j = 1:ncols
      crosscorr[:, i, j] = acf(x[:, i], x[:, j], lags, correlation=correlation, demean=demean)
    end
  end
  crosscorr
end

# Cross-correlation between all pairs of columns of a matrix at a specific lag
acfall{T<:Real}(x::AbstractMatrix{T}, lags::Real; correlation::Bool=true, demean::Bool=true) =
  reshape(acfall(x, lags:lags, correlation=correlation, demean=demean), size(x, 2), size(x, 2))

# Unlike acfall, compute only autocorrelation (not cross-correlation) of matrix columns for range
function acfdiag{T<:Real}(x::AbstractMatrix{T}, lags::Ranges=0:min(size(x, 1)-1, 10log10(size(x, 1)));
  correlation::Bool=true, demean::Bool=true)
  ncols = size(x, 2)

  autocorr = Array(T, length(lags), ncols)
  for i = 1:ncols
    autocorr[:, i] = acf(x[:, i], lags, correlation=correlation, demean=demean)
  end
  autocorr
end

# Unlike acfall, compute only autocorrelation (not cross-correlation) of matrix columns at a specific lag
acfdiag{T<:Real}(x::AbstractMatrix{T}, lags::Real; correlation::Bool=true, demean::Bool=true) =
  acfdiag(x, lags:lags, correlation=correlation, demean=demean)

# acf wrapper (with matrix as input) for range
function acf{T<:Real}(x::AbstractMatrix{T}, lags::Ranges=0:min(size(x, 1)-1, 10log10(size(x, 1)));
  correlation::Bool=true, demean::Bool=true, diag::Bool=false)
  diag ?
    acfdiag(x, lags, correlation=correlation, demean=demean) :
    acfall(x, lags, correlation=correlation, demean=demean)
end

# acf wrapper (with matrix as input) at a specific lag
function acf{T<:Real}(x::AbstractMatrix{T}, lags::Real; correlation::Bool=true, demean::Bool=true, diag::Bool=false)
 diag ?
   acfdiag(x, lags, correlation=correlation, demean=demean) :
   acfall(x, lags, correlation=correlation, demean=demean)
end
