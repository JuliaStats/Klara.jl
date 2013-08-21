###########################################################################
#
#  Vanilla runner : consumes repeatedly a sampler and returns a MCMCChain
#
#
###########################################################################

# import Base.getindex, 
import Base.run

export run

function run(t::MCMCTask; steps::Integer=100, burnin::Integer=0)
  assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
  assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")

  tic() # start timer

  # temporary array to store samples
  samples = fill(NaN, t.model.size, steps-burnin) 

  # sampling loop
  for i in 1:steps
    newprop = consume(t.task)
    i > burnin && (samples[:, i-burnin] = newprop.beta)
  end

  # build the samples dataframe by splitting the 'samples' array
  # into as many columns as there are variables in the model's pmap
  d = DataFrame()
  for (k,v) in t.model.pmap
      col = mapslices(x->Array[x], samples[v.pos:(v.pos+prod(v.dims)-1), :], 2)
      d[string(k)] = col
  end

  MCMCChain((burnin+1):1:steps,
            d,
            DataFrame(),  # TODO, store gradient here, needs to be passed by newprop
            DataFrame(),  # TODO, store diagnostics here, needs to be passed by newprop
            t,
            toq())
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
*(t::Union(MCMCTask, Array{MCMCTask}, MCMCChain, Array{MCMCChain}), i::Range1{Int}) = run(t, steps=i.start+i.len-1, burnin=i.start-1)
