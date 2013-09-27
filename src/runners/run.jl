###########################################################################
#
#  Vanilla runner : consumes repeatedly a sampler and returns a MCMCChain
#
#
###########################################################################

import Base.run
export run

immutable SingleMCRunner <: MCMCRunner
  burnin::Integer
  thinning::Integer
  len::Integer
  r::Range

  function SingleMCRunner(steps::Range{Int})
  r = steps

  burnin = first(r)-1
  thinning = r.step
  len = last(r)

  assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
  assert(len > burnin, "Total MCMC length ($len) should be > to burnin ($burnin)")  
  assert(thinning >= 1, "Thinning ($thinning) should be >= 1") 

  new(burnin, thinning, len, r)
  end
end

SingleMCRunner(steps::Range1{Int}) = SingleMCRunner(first(steps):1:last(steps))

SingleMCRunner(steps::Integer, burnin::Integer=0, thinning::Integer=1) = SingleMCRunner((burnin+1):thinning:steps)

function run( t::MCMCTask; 
              steps::Union(Integer, Range{Int}, Range1{Int})=100, 
              burnin::Integer=0, 
              thinning::Integer=1)
  
  # calculates sampling range 'r' depending on type of 'steps'
  r = isa(steps, Integer) ? ((burnin+1):thinning:steps) : 
          isa(steps, Range1) ? (first(steps):1:last(steps)) : steps

  burnin = first(r)-1
  thinning = r.step
  len = last(r)

  assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
  assert(len > burnin, "Total MCMC length ($len) should be > to burnin ($burnin)")  
  assert(thinning >= 1, "Thinning ($thinning) should be >= 1")  

  tic() # start timer

  # temporary array to store samples
  samples = fill(NaN, t.model.size, length(r))
  gradients = fill(NaN, t.model.size, length(r))
  diags = DataFrame(step=collect(r)) # initialize with column 'step'

  # sampling loop
  j = 1
  for i in 1:len
    newprop = consume(t.task)
    if in(i, r)
      samples[:, j] = newprop.ppars
      if newprop.pgrads != nothing
        gradients[:, j] = newprop.pgrads
      end

      # save diagnostics
      for (k,v) in newprop.diagnostics
        # if diag name not seen before, create column
        if !in(k, colnames(diags))
          diags[string(k)] = DataArray(Array(typeof(v), nrow(diags)), falses(nrow(diags)) )
        end
        
        diags[j, string(k)] = v
      end

      j += 1
    end
  end

  # generate column names for the samples DataFrame
  cn = ASCIIString[]
  for (k,v) in t.model.pmap
      if length(v.dims) == 0 # scalar
        push!(cn, string(k))
      elseif length(v.dims) == 1 # vector
        cn = vcat(cn, ASCIIString[ "$k.$i" for i in 1:v.dims[1] ])
      elseif length(v.dims) == 2 # matrix
        cn = vcat(cn, ASCIIString[ "$k.$i.$j" for i in 1:v.dims[1], j in 1:v.dims[2] ])
      end
  end

  # create Chain
  MCMCChain(r,
            DataFrame(samples', cn),
            DataFrame(gradients', cn),
            diags, 
            t,
            toq())
end

# chain continuation alternate
run(c::MCMCChain; args...) = run(c.task; args...)

# vectorized version of 'run' for arrays of MCMCTasks or MCMCChains
# TODO : use multiple cores if available
function run(t::Union(Array{MCMCTask}, Array{MCMCChain}); args...)
  res = Array(MCMCChain, size(t))
    for i = 1:length(t)
        res[i] = run(t[i]; args...)
    end	
  res
end

# Alternate version with Model and Sampler passed separately
run{M<:MCMCModel, S<:MCMCSampler}(m::Union(M, Vector{M}), s::Union(S, Vector{S}); args...) = run(m * s; args...)

# Syntax shorcut using *
*(t::Union(MCMCTask, Array{MCMCTask}, MCMCChain, Array{MCMCChain}), range::Range{Int}) = run(t, steps=range)
*(t::Union(MCMCTask, Array{MCMCTask}, MCMCChain, Array{MCMCChain}), range::Range1{Int}) = run(t, steps=range)
