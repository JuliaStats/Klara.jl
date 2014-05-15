import Base.mean

export mean

# Mean of MCMCChain
function mean(c::MCMCChain, par::Ranges=1:size(c.samples, 2))
  if c.useAllSamples
    indx = 1:size(c.samples, 1)
  else
    indx = find(c.diagnostics["accept"])
  end
  Float64[mean(c.samples[indx, i]) for i = par]
end

mean(c::MCMCChain, par::Real) = mean(c, par:par)
