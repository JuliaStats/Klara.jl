###########################################################################
#
#  Vanilla runner : consumes repeatedly a sampler and returns a MCMCChain
#
#
###########################################################################

import Base.getindex, Base.run
export run, getindex

function run(t::MCMCTask; steps::Integer=100, burnin::Integer=0)
	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")

	res = MCMCChain({:beta => fill(NaN, t.model.size, steps-burnin)}, t, NaN)

	tic() # start timer

	for i in 1:steps	
		newprop = consume(t.task)
		i > burnin && (res.parameters[:beta][:, i-burnin] = newprop)
	end

	res.runTime = toq()
	res
end

# chain continuation alternate (with default burnin set at zero)
run(c::MCMCChain; steps::Integer=100, burnin::Integer=0) = run(c.task, steps=steps, burnin=burnin)

# vectorized version of 'run' for arrays of MCMCTasks or MCMCChains
function run(t::Union(Array{MCMCTask}, Array{MCMCChain}); steps::Integer=100, burnin::Integer=10)
	res = Array(MCMCChain, size(t))
    for i = 1:length(t)
        res[i] = run(t[i], steps=steps, burnin=burnin)
    end	
	res
end

# alternate version with Model and Sampler passed separately
run{M<:MCMCModel, S<:MCMCSampler}(m::Union(M, Vector{M}), 
	                              s::Union(S, Vector{S}); 
	                              steps::Integer=100, 
	                              burnin::Integer=10) = run(m * s, steps=steps, burnin=burnin)


# syntax shorcut using getIndex
getindex(t::Union(MCMCTask, Array{MCMCTask}, MCMCChain, Array{MCMCChain}), i::Range1{Int}) = 
	run(t, steps=i.start+i.len-1, burnin=i.start-1)

