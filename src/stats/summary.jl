import Base.mean

export mean

# Mean of MCMCChain
function mean(c::MCMCChain)
  npars = size(c.samples, 2)
  means = Array(Float64, npars)

  for i = 1:npars
    means[i] = mean(c.samples[:, i])
  end

  return means
end
