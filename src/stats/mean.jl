# Mean of MCChain

mean(c::MCChain, par::Range=1:size(c.samples, 2)) = Float64[mean(c.samples[:, i]) for i = par]

mean(c::MCChain, par::Real) = mean(c, par:par)
