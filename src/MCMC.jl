module MCMC

import Base.*, Base.show
export *
export MCMCModel

type MCMCModel
	eval::Function
	pmap::Dict
	size::Integer
	init::Vector{Float64}
end

type MCMCTask
	task::Task
	model::MCMCModel
end


type MCMCChain
	parameters::Dict
	task::MCMCTask
	runTime::Float64
end

function show(io::IO, res::MCMCChain)
	local samples = 0
	for v in keys(res.parameters)
		print("$v$(size(res.parameters[v])[1:end-1]) ")
		samples = size(res.parameters[v])[end]
	end 
	print("by $samples samples, ")
	println("$(round(res.runTime,1)) sec.")
end


abstract MCMCSampler


include("samplers/RWM.jl")
include("runners/run.jl")


reset(t::Task, x) = t.storage[:reset](x)

#  Definition of * as a shortcut operator for model and sampler combination 
*{M<:MCMCModel, S<:MCMCSampler}(m::M, s::S) = spinTask(m, s)
*{M<:MCMCModel, S<:MCMCSampler}(m::Array{M}, s::S) = map((me) -> spinTask(me, s), m)
*{M<:MCMCModel, S<:MCMCSampler}(m::M, s::Array{S}) = map((se) -> spinTask(m, se), s)

end