import Base.mean, DataFrames.describe

export mean, acceptance, describe

# Mean of MCMCChain
mean(c::MCMCChain, pars::Ranges=1:ncol(c.samples)) = Float64[mean(c.samples[:, i]) for i = pars]

mean(c::MCMCChain, par::Real) = mean(c, par:par)

# Compute acceptance or rejection rate of MCMCChain
function acceptance(c::MCMCChain; lags::Ranges=1:nrow(c.samples), reject::Bool=false)
  rlen = lags.len
  assert (lags[end] <= nrow(c.samples), "Range of acceptance rate not within post-burnin range of MCMC chain")

  if reject
    return (rlen-sum(c.diagnostics[lags, "accept"]))*100/rlen
  else
    return sum(c.diagnostics[lags, "accept"])*100/rlen
  end
end

# TODO 1: Compute MCMC quantiles based on
# Flegal J.M, Galin L.J, Neath R.C. Markov Chain Monte Carlo Estimation of Quantiles. arXiv, 2013
# TODO 2: Include these MCMC estimates of quantiles in describe()
# TODO 3: Improve describe() definition - it may be helpful to split MCMCChain to MCMCFrame and MCMCVector
# TODO 4: After TODO 3, it will become possible to call var(MCMCVector) and include the output in describe()

# describe() provides summary statistics for MCMCChain objects, similarly to dataframes' describe
describe(c::MCMCChain) = describe(STDOUT, c)

function describe(io, c::MCMCChain)
  for i in 1:ncol(c.samples)
    col = c.samples[i]
    println(io, colnames(c.samples)[i])

    if all(isna(col))
      println(col, " * All NA * ")
        return
    end

    filtered = float(removeNA(col))
    qs = quantile(filtered, [0, 1])
    nrows = nrow(c.samples)
    varimse = mcvar_imse(filtered)
    variid = var(filtered)/nrows
    mcerror = sqrt(varimse)
    ss = nrows*variid./varimse
    act = varimse./variid

    statNames = ["Min", "Mean", "Max", "MC Error", "ESS", "AC Time"]
    statVals = [qs[1], mean(filtered), qs[2], mcerror, ss, act]
    for i = 1:6
        println(io, string(rpad(statNames[i], 10, " "), " ", string(statVals[i])))
    end
    nas = sum(isna(col))
    println(io, "NAs        $nas")
    println(io, "NA%        $(round(nas*100/length(col), 2))%")

    println(io, )
  end
end
