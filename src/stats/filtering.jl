### Default filtering method selects all MCMC samples
select(c::MCChain, s::MCSampler) = trues(size(c.samples, 1))

### Sampler-specific selectors
# select(c::MCChain, s::ARS) = c.diagnotics["accept"]
