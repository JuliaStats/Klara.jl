import Base.var, Base.std
export var, std

# Variance of MCMCChain
function var_iid(c::MCMCChain)
  nsamples, npars = size(c.samples)
  Float64[var(c.samples[:, i]) for i = 1:npars]/nsamples
end

# Standard deviation of MCMCChain
std_iid(c::MCMCChain) = sqrt(var_iid(c.samples))

# Function for estimating Monte Carlo error using batch means, see for instance
# Flegal J.M, Jones, G.L. Batch Means and Spectral Variance Estimators in Markov chain Monte Carlo. Annals of
# Statistics, 2010, 38 (2), pp 1034-1070
function var_bm(c::MCMCChain; batchlen::Int=100)
  nsamples, npars = size(c.samples)
  nbatches = div(nsamples, batchlen)  
  assert(nbatches > 1, "Choose batch size such that the number of batches is greather than one")
  nbsamples = nbatches*batchlen

  batchmeans, variance = Array(Float64, nbatches), Array(Float64, npars)

  for i = 1:npars
    batchmeans = Float64[mean(c.samples[((j-1)*batchlen+1):(j*batchlen), i]) for j = 1:nbatches]
    variance[i] = batchlen*var(batchmeans)/nbsamples
  end

  return variance
end

# Monte Carlo standard error using batch means
std_bm(c::MCMCChain; batchlen::Int=100) = sqrt(var_bm(c, batchlen=batchlen))

# Function for estimating the variance of a single MCMC chain using the initial monotone sequence estimator (IMSE), see
# Geyer C.J. Practical Markov Chain Monte Carlo. Statistical Science, 1992, 7 (4), pp 473-483
function var_imse(c::MCMCChain; maxlag::Int=nrow(c.samples)-1)
  nsamples, npars = size(c.samples)

  k = convert(Int, floor((maxlag-1)/2))
  
  # Preallocate memory
  acv = Array(Float64, maxlag+1, npars)
  g = Array(Float64, k+1, npars)
  m = (k+1)*ones(npars)

  # Calculate empirical autocovariance
  for i = 1:npars
    acv[:, i] = acf(c.samples[:, i], 0:maxlag, correlation=false)
  end

  # Calculate \hat{G}_{n, m} from the autocovariances, see pp. 477 in Geyer
  for i = 1:npars
    for j = 0:k
      g[j+1, i] = acv[2*j+1, i]+acv[2*j+2, i]
      if g[j+1, i] <= 0
        m[i] = j
        break
      end
    end
  end

  # Create the monotone sequence of g
  for i = 1:npars
    if m[i] > 1
      for j = 2:m[i]
        if g[j, i] > g[j-1, i]
          g[j, i] = g[j-1, i]
        end
      end
    end
  end

  # Calculate the initial monotone sequence estimator
  return Float64[-acv[1, i]+2*sum(g[1:m[i], i]) for i = 1:npars]/nsamples
end

# Standard deviation using Geyer's IMSE
std_imse(c::MCMCChain; maxlag::Int=nrow(c.samples)-1) = sqrt(var_imse(c, maxlag=maxlag))

# Function for estimating the variance of a single MCMC chain using the initial positive sequence estimator (IPSE), see
# Geyer C.J. Practical Markov Chain Monte Carlo. Statistical Science, 1992, 7 (4), pp 473-483
function var_ipse(c::MCMCChain; maxlag::Int=nrow(c.samples)-1)
  nsamples, npars = size(c.samples)

  k = convert(Int, floor((maxlag-1)/2))
  
  # Preallocate memory
  acv = Array(Float64, maxlag+1, npars)
  g = Array(Float64, k+1, npars)
  m = (k+1)*ones(npars)
  variance = Array(Float64, npars)

  # Calculate empirical autocovariance
  for i = 1:npars
    acv[:, i] = acf(c.samples[:, i], 0:maxlag, correlation=false)
  end

  # Calculate \hat{G}_{n, m} from the autocovariances, see pp. 477 in Geyer
  for i = 1:npars
    for j = 0:k
      g[j+1, i] = acv[2*j+1, i]+acv[2*j+2, i]
      if g[j+1, i] <= 0
        m[i] = j
        break
      end
    end
  end

  # Calculate the initial positive sequence estimator  
  return Float64[-acv[1, i]+2*sum(g[1:m[i], i]) for i = 1:npars]/nsamples
end

# Standard deviation using Geyer's IPSE
std_ipse(c::MCMCChain; maxlag::Int=nrow(c.samples)-1) = sqrt(var_ipse(c, maxlag=maxlag))

# Wrapper function for computing MCMC variance using various approaches
vtypes = (:bm, :iid, :imse, :ipse)

function var(c::MCMCChain; vtype::Symbol=:imse, args...)
  assert(in(vtype, vtypes), "Unknown variance type $vtype")

  if vtype == :bm
    return var_bm(c; args...) 
  elseif vtype == :iid
    return var_iid(c)   
  elseif vtype == :imse
    return var_imse(c; args...)
  elseif vtype == :ipse
    return var_ipse(c; args...)
  end
end

# Wrapper function for computing Monte Carlo (standard) error using various approaches
function std(c::MCMCChain; vtype::Symbol=:imse, args...)
  assert(in(vtype, vtypes), "Unknown standard error type $vtype")

  if vtype == :bm
    return std_bm(c; args...)
  elseif vtype == :iid
    return std_iid(c)
  elseif vtype == :imse
    return std_imse(c; args...)
  elseif vtype == :ipse
    return std_ipse(c; args...)   
  end
end
