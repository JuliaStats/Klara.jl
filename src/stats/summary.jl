### Compute acceptance or rejection rate of MCChain

function acceptance(c::MCChain; lags::Ranges=1:size(c.samples, 1), reject::Bool=false)
  rlen = length(lags)
  @assert lags[end] <= size(c.samples, 1) "Range of acceptance rate not within post-burnin range of MCmc chain"

  if reject
    return (rlen-sum(c.diagnostics["accept"][lags]))*100/rlen
  else
    return sum(c.diagnostics["accept"][lags])*100/rlen
  end
end

### describe() provides summary statistics for MCChain objects

describe(c::MCChain) = describe(STDOUT, c)

function describe(io, c::MCChain)
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
