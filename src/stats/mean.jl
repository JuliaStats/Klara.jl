import Base.mean

export mean, mean_rb

# Mean of MCMCChain
mean(c::MCMCChain, par::Ranges=1:size(c.samples, 2)) = Float64[mean(c.samples[:, i]) for i = par]

mean(c::MCMCChain, par::Real) = mean(c, par:par)

# Mean of MCMCChain using Rao-Blackwell samples
function mean_rb_hmc(c::MCMCChain, par::Ranges=1:size(c.samples, 2))
  nsamples, npars, nleaps = size(c.samples, 1), length(par), length(c.diagnostics["leaps"][1])-1

  w = Array(Float64, nsamples, nleaps)
  sums = Array(Float64, nsamples, npars)

  for i = 1:nsamples
    for j = 1:nleaps
      w[i, j] = exp(c.diagnostics["leaps"][i][1].H-c.diagnostics["leaps"][i][j+1].H)
    end
  end

  for i = 1:nsamples
    for j = 1:npars
      s = c.samples[i, j]
      for k = 1:nleaps
        s += w[i, k]*c.diagnostics["leaps"][i][k+1].pars[j]
      end

      sums[i, j] = s/(nleaps+1)
    end
  end

  mean(sums, 1)
end

mean_rb_hmc(c::MCMCChain, par::Real) = mean_rb_hmc(c, par:par)

function mean_rb(c::MCMCChain, p::Union(Real, Ranges)=1:size(c.samples, 2), s::Symbol=:hmc)
  if s == :hmc
    mean_rb_hmc(c, p)
  end
end
