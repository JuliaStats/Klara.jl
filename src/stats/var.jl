import Base.var, Base.std
export var, var_imse, std, std_imse

# Variance of MCMCChain
function var_iid(c::MCMCChain)
  nsamples, npars = size(c.samples)
  variance = Array(Float64, npars)

  for i = 1:npars
    variance[i] = var(c.samples[:, i])
  end

  return variance/nsamples
end

# Standard deviation of MCMCChain
std_iid(c::MCMCChain) = sqrt(var_iid(c.samples))

# Function for estimating the variance of a single MCMC chain using the initial monotone sequence estimator (IMSE), see
# Geyer C.J. Practical Markov Chain Monte Carlo. Statistical Science, 1992, 7 (4), pp 473-483
function var_imse(c::MCMCChain, maxlag::Int)
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
  for i = 1:npars
    variance[i] = -acv[1, i]+2*sum(g[1:m[i], i])
  end
  
  return variance/nsamples
end

var_imse(c::MCMCChain) = var_imse(c, size(c.samples, 1)-1)

# Standard deviation using Geyer's IMSE
std_imse(c::MCMCChain, maxlag::Int) = sqrt(var_imse(c, maxlag))

std_imse(c::MCMCChain) = sqrt(var_imse(c, size(c.samples, 1)-1))

# Function for estimating the variance of a single MCMC chain using the initial positive sequence estimator (IPSE), see
# Geyer C.J. Practical Markov Chain Monte Carlo. Statistical Science, 1992, 7 (4), pp 473-483
function var_ipse(c::MCMCChain, maxlag::Int)
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
  for i = 1:npars
    variance[i] = -acv[1, i]+2*sum(g[1:m[i], i])
  end
  
  return variance/nsamples
end

var_ipse(c::MCMCChain) = var_ipse(c, size(c.samples, 1)-1)

# Standard deviation using Geyer's IPSE
std_ipse(c::MCMCChain, maxlag::Int) = sqrt(var_ipse(c, maxlag))

std_ipse(c::MCMCChain) = sqrt(var_ipse(c, size(c.samples, 1)-1))

# Wrapper function for computing MCMC variance using various approaches
vtypes = (:iid, :imse, :ipse)

function var(c::MCMCChain; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown variance type $vtype")

  if vtype == :iid
    return var_iid(c)
  elseif vtype == :imse
    return var_imse(c)
  elseif vtype == :ipse
    return var_ipse(c)
  end
end

# Wrapper function for computing Monte Carlo (standard) error using various approaches
function std(c::MCMCChain; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown standard error type $vtype")

  if vtype == :iid
    return std_iid(c)
  elseif vtype == :imse
    return std_imse(c)
   elseif vtype == :ipse
    return std_ipse(c)   
  end
end
