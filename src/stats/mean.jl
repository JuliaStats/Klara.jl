import Base.mean

export mean

# Mean of MCMCChain
mean(c::MCMCChain, par::Ranges=1:size(c.samples, 2)) = Float64[mean(c.samples[:, i]) for i = par]

mean(c::MCMCChain, par::Real) = mean(c, par:par)
