export acceptance, describe

# Compute acceptance or rejection rate of MCMCChain
function acceptance(c::MCMCChain; lags::Ranges=1:size(c.samples, 1), reject::Bool=false)
  rlen = length(lags)
  @assert lags[end] <= size(c.samples, 1) "Range of acceptance rate not within post-burnin range of MCMC chain"

  if reject
    return (rlen-sum(c.diagnostics["accept"][lags]))*100/rlen
  else
    return sum(c.diagnostics["accept"][lags])*100/rlen
  end
end

# TODO 1: Compute MCMC quantiles based on
# Flegal J.M, Galin L.J, Neath R.C. Markov Chain Monte Carlo Estimation of Quantiles. arXiv, 2013
# TODO 2: Include these MCMC estimates of quantiles in describe()

# describe() provides summary statistics for MCMCChain objects
describe(c::MCMCChain) = describe(STDOUT, c)

function describe(io, c::MCMCChain)
  nsamples, npars = size(c.samples)
  for i in 1:npars

    col = c.samples[:, i]
    println(io, "Parameter $i")

    if sum(isnan(col)) != 0
      println(col, "Monte Carlo chain for parameter $i contains NaNs, which are not supported.")
      return
    end

    qs = quantile(col, [0, 1])
    varimse = mcvar_imse(col)
    variid = var(col)/nsamples
    mcerror = sqrt(varimse)
    ss = nsamples*variid./varimse
    act = varimse./variid

    statNames = ["Min", "Mean", "Max", "MC Error", "ESS", "AC Time"]
    statVals = [qs[1], mean(col), qs[2], mcerror, ss, act]
    for i = 1:6
        println(io, string(rpad(statNames[i], 10, " "), " ", string(statVals[i])))
    end

    println(io, )
  end
end
