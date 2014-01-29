##########################################################################################
#
#    Log likelihood accumulator : type definition and derivation rules
#
##########################################################################################

# this makes the model function easier to generate compared to a Float64
#   - embeds the error throwing when log-likelihood reaches -Inf
#   - calculates the sum when logpdf() returns an Array
type OutOfSupportError <: Exception ; end

immutable LLAcc
	val::Float64
	function LLAcc(x::Real)
		isfinite(x) || throw(OutOfSupportError())
		new(x)
	end
end
+(ll::LLAcc, x::Real)           = LLAcc(ll.val + x)
+(ll::LLAcc, x::Array{Float64}) = LLAcc(ll.val + sum(x))

declareType(LLAcc, :LLAcc) # declares new type to Autodiff

####### derivation rules  ############
# (note : only additions are possible with LLAcc type )
@deriv_rule getfield(x::LLAcc, f      )      x     dx1 = ds

@deriv_rule +(x::LLAcc, y      )             x     dx1 += ds1
@deriv_rule +(x::LLAcc, y::Real)             y     dy += ds1
@deriv_rule +(x::LLAcc, y::AbstractArray)    y     for i in 1:length(y) ; dy[i] += ds1 ; end

