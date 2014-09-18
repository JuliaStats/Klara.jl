#################################################################
#
#    Definition of MCMCModel types 
#
#################################################################

function ispartition(m::Dict, n::Int)
	c = zeros(n)
	for v in values(m)
		c[v[1]:(v[1]+prod(v[2])-1)] = c[v[1]:(v[1]+prod(v[2])-1)] .+ 1
	end
	all(c .== 1.)
end

#### misc functions common to all models  ####
hasgradient{M<:MCMCModel}(m::M) = m.evalg != nothing
hastensor{M<:MCMCModel}(m::M) = m.evalt != nothing
hasdtensor{M<:MCMCModel}(m::M) = m.evaldt != nothing

#### User-facing model creation function  ####

# Currently only a "likelihood" model type makes sense
# Left as is in case other kind of models come up
function model(f::Union(Function, Distribution, Expr); mtype="likelihood", args...)
	if mtype == "likelihood"
		return MCMCLikelihoodModel(f; args...)
	elseif mtype == "whatever"
	else
	end
end
