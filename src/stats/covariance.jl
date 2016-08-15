### Compute empirical covariance matrix recursively

function covariance!(
  C::RealMatrix,
  lastC::RealMatrix,
  k::Integer,
  x::RealVector,
  lastmean::RealVector,
  secondlastmean::RealVector
)
  C[:, :] = (k-1)*lastC
  BLAS.ger!(1.0, x, x, C)
  BLAS.ger!(1.0, -(k+1)*lastmean, lastmean, C)
  BLAS.ger!(1.0, k*secondlastmean, secondlastmean, C)
  C[:, :] /= k
end
