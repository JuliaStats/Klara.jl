import Base.var, Base.std
export var, var_imse, std, std_imse

# Variance for dataframe argument
function var(x::DataFrame)
  nrows, ncols = size(x)
  variance = Array(Float64, ncols)

  for i = 1:ncols
    variance[i] = var(x[:, i])
  end

  variance
end

# Standard deviation for dataframe argument
std(x::DataFrame) = sqrt(var(x))

# Function for estimating the variance of a single MCMC chain using the initial monotone sequence estimator (IMSE) of
# Geyer, see Practical Markov Chain Monte Carlo, C. J. Geyer, Statistical Science, Vol. 7, No. 4. (1992), pp. 473-483
function var_imse(mcmc::DataFrame, maxlag::Int)
  nsamples, npars = size(mcmc)

  k = convert(Int, floor((maxlag-1)/2))
  
  # Preallocate memory
  acv = Array(Float64, maxlag+1, npars)
  g = Array(Float64, k+1, npars)
  m = (k+1)*ones(npars)
  variance = Array(Float64, npars)

  # Calculate empirical autocovariance
  for i = 1:npars
    acv[:, i] = acf(mcmc[:, i], 0:maxlag, correlation=false)
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

var_imse(mcmc::DataFrame) = var_imse(mcmc::DataFrame, size(mcmc, 1)-1)

# Standard deviation using Geyer's IMSE
std_imse(mcmc::DataFrame, maxlag::Int) = sqrt(var_imse(mcmc::DataFrame, maxlag))

std_imse(mcmc::DataFrame) = sqrt(var_imse(mcmc::DataFrame, size(mcmc, 1)-1))

# Wrapper function for computing MCMC variance using various approaches
vtypes = (:imse, :ipse)

function var(mcmc::DataFrame; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown variance type $vtype")

  if vtype == :imse
    return var_imse(mcmc)
  end
end

# Wrapper function for computing Monte Carlo (standard) error using various approaches
function std(mcmc::DataFrame; vtype::Symbol=:imse)
  assert(in(vtype, vtypes), "Unknown standard error type $vtype")

  if vtype == :imse
    return std_imse(mcmc)
  end
end
