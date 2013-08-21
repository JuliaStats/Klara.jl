###########################################################################
#
#  Vanilla runner : consumes repeatedly a sampler and returns a MCMCChain
#
#
###########################################################################

# import Base.getindex, 
import Base.run

export run  #, getindex

function run(t::MCMCTask; nsteps::Integer=100, burnin::Integer=0, thinning::Integer=1)
  local len = burnin+nsteps*thinning
  
  assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
  assert(len > burnin, "Total MCMC length ($len) should be > to burnin ($burnin)")  

  res = MCMCChain((burnin+1):thinning:len, DataFrame(t.model.size, nsteps), t)

  tic() # start timer

  for i in 1:len
    newprop = consume(t.task)
    
    if i > burnin && (i % thinning == 0)
      res.samples[:, div(i-burnin, thinning)] = newprop.beta
    end
  end

  res.runTime = toq()
  res
end

# chain continuation alternate
run(c::MCMCChain; args...) = run(c.task; args...)

# vectorized version of 'run' for arrays of MCMCTasks or MCMCChains
function run(t::Union(Array{MCMCTask}, Array{MCMCChain}); args...)
  res = Array(MCMCChain, size(t))
    for i = 1:length(t)
        res[i] = run(t[i]; args...)
    end	
  res
end

# alternate version with Model and Sampler passed separately
run{M<:MCMCModel, S<:MCMCSampler}(m::Union(M, Vector{M}), s::Union(S, Vector{S}); args...) = run(m * s; args...)


# syntax shorcut using *
# getindex(t::Union(MCMCTask, Array{MCMCTask}, MCMCChain, Array{MCMCChain}), i::Range1{Int}) = 
# 	run(t, nsteps=i.start+i.len-1, burnin=i.start-1)
*(t::Union(MCMCTask, Array{MCMCTask}, MCMCChain, Array{MCMCChain}), range::Range{Int}) = run(t, nsteps=range.len, burnin=range.start-1, thinning=range.step)
*(t::Union(MCMCTask, Array{MCMCTask}, MCMCChain, Array{MCMCChain}), range::Range1{Int}) = run(t, nsteps=range.len, burnin=range.start-1)
