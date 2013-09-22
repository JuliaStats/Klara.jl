import Base.mean

export mean

# Mean of MCMCChain
mean(c::MCMCChain) = [mean(c.samples[:, i]) for i = 1:size(c.samples, 2)]
