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

  # generate column names
  cn = []
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
  MCMCChain((burnin+1):1:steps,
            DataFrame(samples', cn),
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
