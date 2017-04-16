### Compute empirical covariance matrix recursively

covariance(lastC::Real, k::Integer, x::Real, lastmean::Real, secondlastmean::Real) =
  ((k-1)*lastC+abs2(x)-(k+1)*abs2(lastmean)+k*abs2(secondlastmean))/k

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
