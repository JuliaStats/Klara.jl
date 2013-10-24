import Base.mean

export mean, mean_rb

# Mean of MCMCChain
mean(c::MCMCChain, pars::Ranges=1:ncol(c.samples)) = Float64[mean(c.samples[:, i]) for i = pars]

mean(c::MCMCChain, par::Real) = mean(c, par:par)

# Mean of MCMCChain using Rao-Blackwell samples
function mean_rb_hmc(c::MCMCChain, pars::Ranges=1:ncol(c.samples))
  nsamples, npars, nleaps = nrow(c.samples), length(pars), length(c.diagnostics["leaps"][1])-1

  w = Array(Float64, nsamples, nleaps)
  sums = Array(Float64, nsamples, npars)

  for i = 1:nsamples
    for j = 1:nleaps
      w[i, j] = c.diagnostics["leaps"][i][1].H-c.diagnostics["leaps"][i][j+1].H
    end
  end

  for i = 1:nsamples
    for j = 1:npars
      s = 0
      for k = 1:nleaps
        s += w[i, k]*c.diagnostics["leaps"][i][k+1].pars[j]
      end

      sums[i, j] = c.samples[i, j]+s
    end
  end

  sum(sums, 1)/(nsamples*(nleaps+1))
end

mean_rb_hmc(c::MCMCChain, par::Real) = mean_rb_hmc(c, par:par)

function mean_rb(c::MCMCChain, p::Union(Real, Ranges)=1:ncol(c.samples), s::Symbol=:hmc)
  if s == :hmc
    mean_rb_hmc(c, p)
  end
end
