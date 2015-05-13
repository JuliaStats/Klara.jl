### mcvar and mcse stand for Monte Carlo variance and Monte Carlo error respectively

### Monte Carlo variance (erroneously) assuming IID samples

mcvar_iid(x::Vector{Float64}) = var(x)/length(x)

mcvar_iid(x::Matrix{Float64}, pars::Range=1:size(x, 2)) = Float64[mcvar_iid(x[:, pars[i]]) for i = 1:length(pars)]

mcvar_iid(x::Matrix{Float64}, par::Real) = mcvar_iid(x, par:par)

mcvar_iid(c::MCChain, pars::Range=1:size(c.samples, 2)) = mvcvar_iid(c.samples, pars)

mcvar_iid(c::MCChain, par::Real) = mcvar_iid(c, par:par)

### Monte Carlo standard error (erroneously) assuming IID samples

msce_iid(x::Vector{Float64}) = sqrt(mcvar_iid(x))

msce_iid(x::Matrix{Float64}, pars::Range=1:size(x, 2)) = Float64[mcse_iid(x[:, pars[i]]) for i = 1:length(pars)]

msce_iid(x::Matrix{Float64}, par::Real) = msce_iid(x, par:par)

msce_iid(c::MCChain, pars::Range=1:size(c.samples, 2)) = msce_iid(c.samples, pars)

msce_iid(c::MCChain, par::Real) = msce_iid(c, par:par)

### Function for estimating Monte Carlo variance using batch means, see for instance Flegal J.M, Jones, G.L. Batch Means
### and Spectral Variance Estimators in Markov chain Monte Carlo. Annals of Statistics, 2010, 38 (2), pp 1034-1070

function mcvar_bm(x::Vector{Float64}; batchlen::Int=100)
  nbatches = div(length(x), batchlen)
  @assert nbatches > 1 "Choose batch size such that the number of batches is greather than one"
  nbsamples = nbatches*batchlen
  batchmeans = Float64[mean(x[((j-1)*batchlen+1):(j*batchlen)]) for j = 1:nbatches]
  return batchlen*var(batchmeans)/nbsamples
end

mcvar_bm(x::Matrix{Float64}, pars::Range=1:size(x, 2); batchlen::Int=100) =
  Float64[mcvar_bm(x[:, pars[i]]; batchlen=batchlen) for i = 1:length(pars)]

mcvar_bm(x::Matrix{Float64}, par::Real; batchlen::Int=100) = mcvar_bm(x, par:par; batchlen=batchlen)

mcvar_bm(c::MCChain, pars::Range=1:size(c.samples, 2); batchlen::Int=100) =
  mcvar_bm(c.samples, pars; batchlen=batchlen)

mcvar_bm(c::MCChain, par::Real; batchlen::Int=100) = mcvar_bm(c, par:par; batchlen=batchlen)

### Monte Carlo standard error using batch means

mcse_bm(x::Vector{Float64}; batchlen::Int=100) = sqrt(mcvar_bm(x; batchlen=batchlen))

mcse_bm(x::Matrix{Float64}, pars::Range=1:size(x, 2); batchlen::Int=100) =
  Float64[mcse_bm(x[:, pars[i]]; batchlen=batchlen) for i = 1:length(pars)]

mcse_bm(x::Matrix{Float64}, par::Real; batchlen::Int=100) = mcse_bm(x, par:par; batchlen=batchlen)

mcse_bm(c::MCChain, pars::Range=1:size(c.samples, 2); batchlen::Int=100) =
  mcse_bm(c.samples, pars; batchlen=batchlen)

mcse_bm(c::MCChain, par::Real; batchlen::Int=100) = mcse_bm(c, par:par; batchlen=batchlen)

### Function for estimating the variance of a single MC chain using the initial monotone sequence estimator (IMSE), see
### Geyer C.J. Practical Markov Chain Monte Carlo. Statistical Science, 1992, 7 (4), pp 473-483

function mcvar_imse(x::Vector{Float64}; maxlag::Int=length(x)-1)
  k = convert(Int, floor((maxlag-1)/2))
  m = k+1

  # Preallocate memory
  g = Array(Float64, k+1)

  # Calculate empirical autocovariance
  acv = autocov(x, 0:maxlag)

  # Calculate \hat{G}_{n, m} from the autocovariances, see pp. 477 in Geyer
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
  return (-acv[1]+2*sum(g[1:m]))/length(x)
end

mcvar_imse(x::Matrix{Float64}, pars::Range=1:size(x, 2); maxlag::Int=size(x, 1)-1) =
  Float64[mcvar_imse(x[:, pars[i]]; maxlag=maxlag) for i = 1:length(pars)]

mcvar_imse(x::Matrix{Float64}, par::Real; maxlag::Int=size(x, 1)-1) = mcvar_imse(x, par:par; maxlag=maxlag)

mcvar_imse(c::MCChain, pars::Range=1:size(c.samples, 2); maxlag::Int=size(c.samples, 1)-1) =
  mcvar_imse(c.samples, pars; maxlag=maxlag)

mcvar_imse(c::MCChain, par::Real; maxlag::Int=size(c.samples, 1)-1) = mcvar_imse(c, par:par; maxlag=maxlag)

### Monte Carlo standard error using Geyer's IMSE

mcse_imse(x::Vector{Float64}; maxlag::Int=length(x)-1) = sqrt(mcvar_imse(x; maxlag=maxlag))

mcse_imse(x::Matrix{Float64}, pars::Range=1:size(x, 2); maxlag::Int=size(x, 1)-1) =
  Float64[mcse_imse(x[:, pars[i]]; maxlag=maxlag) for i = 1:length(pars)]

mcse_imse(c::MCChain, pars::Range=1:size(c.samples, 2); maxlag::Int=size(c.samples, 1)-1) =
  mcse_imse(c.samples, pars; maxlag=maxlag)

mcse_imse(c::MCChain, par::Real; maxlag::Int=size(c.samples, 1)-1) = mcse_imse(c, par:par; maxlag=maxlag)

### Function for estimating the variance of a single MC chain using the initial positive sequence estimator (IPSE), see
### Geyer C.J. Practical Markov Chain Monte Carlo. Statistical Science, 1992, 7 (4), pp 473-483

function mcvar_ipse(x::Vector{Float64}; maxlag::Int=length(x)-1)
  k = convert(Int, floor((maxlag-1)/2))

  # Preallocate memory
  g = Array(Float64, k+1)
  m = k+1

  # Calculate empirical autocovariance
  acv = autocov(x, 0:maxlag)

  # Calculate \hat{G}_{n, m} from the autocovariances, see pp. 477 in Geyer
  for j = 0:k
    g[j+1] = acv[2*j+1]+acv[2*j+2]
    if g[j+1] <= 0
      m = j
      break
    end
  end

  # Calculate the initial monotone sequence estimator
  return (-acv[1]+2*sum(g[1:m]))/length(x)
end

mcvar_ipse(x::Matrix{Float64}, pars::Range=1:size(x, 2); maxlag::Int=size(x, 1)-1) =
  Float64[mcvar_ipse(x[:, pars[i]]; maxlag=maxlag) for i = 1:length(pars)]

mcvar_ipse(x::Matrix{Float64}, par::Real; maxlag::Int=size(x, 1)-1) = mcvar_ipse(x, par:par; maxlag=maxlag)

mcvar_ipse(c::MCChain, pars::Range=1:size(c.samples, 2); maxlag::Int=size(c.samples, 1)-1) =
  mcvar_ipse(c.samples, pars; maxlag=maxlag)

mcvar_ipse(c::MCChain, par::Real; maxlag::Int=size(c.samples, 1)-1) = mcvar_ipse(c, par:par; maxlag=maxlag)

### Monte Carlo standard error using Geyer's IMSE

mcse_ipse(x::Vector{Float64}; maxlag::Int=length(x)-1) = sqrt(mcvar_ipse(x; maxlag=maxlag))

mcse_ipse(x::Matrix{Float64}, pars::Range=1:size(x, 2); maxlag::Int=size(x, 1)-1) =
  Float64[mcse_ipse(x[:, pars[i]]; maxlag=maxlag) for i = 1:length(pars)]

mcse_ipse(c::MCChain, pars::Range=1:size(c.samples, 2); maxlag::Int=size(c.samples, 1)-1) =
  mcse_ipse(c.samples, pars; maxlag=maxlag)

mcse_ipse(c::MCChain, par::Real; maxlag::Int=size(c.samples, 1)-1) = mcse_ipse(c, par:par; maxlag=maxlag)

### Wrapper function for computing MC variance using various approaches

vtypes = (:bm, :iid, :imse, :ipse)

function mcvar(x::Vector{Float64}; vtype::Symbol=:imse, args...)
  @assert in(vtype, vtypes) "Unknown variance type $vtype"

  if vtype == :bm
    return mcvar_bm(x; args...)
  elseif vtype == :iid
    return mcvar_iid(x)
  elseif vtype == :imse
    return mcvar_imse(x; args...)
  elseif vtype == :ipse
    return mcvar_ipse(x; args...)
  end
end

mcvar(x::Matrix{Float64}, pars::Range=1:size(x, 2); vtype::Symbol=:imse, args...) =
  Float64[mcvar(x[:, pars[i]]; vtype=vtype, args...) for i = 1:length(pars)]

mcvar(x::Matrix{Float64}, par::Real; vtype::Symbol=:imse, args...) = mcvar(x, par:par; vtype=vtype, args...)

mcvar(c::MCChain, pars::Range=1:size(c.samples, 2); vtype::Symbol=:imse, args...) =
  mcvar(c.samples, pars; vtype=vtype, args...)

mcvar(c::MCChain, par::Real; vtype::Symbol=:imse, args...) = mcvar(c, par:par; vtype=vtype, args...)

### Wrapper function for computing Monte Carlo (standard) error using various approaches

function mcse(x::Vector{Float64}; vtype::Symbol=:imse, args...)
  @assert in(vtype, vtypes) "Unknown stdiance type $vtype"

  if vtype == :bm
    return mcse_bm(x; args...)
  elseif vtype == :iid
    return mcse_iid(x)
  elseif vtype == :imse
    return mcse_imse(x; args...)
  elseif vtype == :ipse
    return mcse_ipse(x; args...)
  end
end

mcse(x::Matrix{Float64}, pars::Range=1:size(x, 2); vtype::Symbol=:imse, args...) =
  Float64[mcse(x[:, pars[i]]; vtype=vtype, args...) for i = 1:length(pars)]

mcse(x::Matrix{Float64}, par::Real; vtype::Symbol=:imse, args...) = mcse(x, par:par; vtype=vtype, args...)

mcse(c::MCChain, pars::Range=1:size(c.samples, 2); vtype::Symbol=:imse, args...) =
  mcse(c.samples, pars; vtype=vtype, args...)

mcse(c::MCChain, par::Real; vtype::Symbol=:imse, args...) = mcse(c, par:par; vtype=vtype, args...)
