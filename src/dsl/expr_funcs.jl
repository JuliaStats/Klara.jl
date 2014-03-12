##########################################################################
#
#    Misc expression manipulation function 
#
##########################################################################

#### translates ~ into regular syntax
function translate(ex::Expr)
	if ex.head == :block 
		return Expr(:block, translate(ex.args)...)

	elseif ex.head == :call && ex.args[1] == :~
		length(ex.args) == 3 || error("Syntax error in ($ex)")
		# isa(ex.args[3], Expr) || error("Syntax error in ($ex)")
		# ex.args[3].head == :call || error("Syntax error in ($ex)")

		ex2 = ex.args[3]
		if isa(ex2, Expr) && length(ex2.args)==2 && ex2.args[1] == :+   #  ~+  (right censoring) statement
			return :( $ACC_SYM += logccdf( $(ex2.args[2]), $(ex.args[2]) ) )

		elseif isa(ex2, Expr) && length(ex2.args)==2 && ex2.args[1] == :-  #  ~-  (left censoring) statement			
			return :( $ACC_SYM += logcdf( $(ex2.args[2]), $(ex.args[2]) ) )

		elseif isa(ex2, Expr) || isa(ex2, Symbol)   # ~ statement
			return :( $ACC_SYM += logpdf( $(ex2), $(ex.args[2]) ) )

		else
			error("Syntax error in ($ex)")

		end
	else
		return ex
	end
end
translate(ex::Vector) = map(translate, ex)
translate(ex::Any) = ex

### creates mapping statements from Vector{Float64} to model parameter variables
function vec2var(;init...)
	ex = Expr[]
	pos = 1
	for (v,i) in init
		sz = size(i)
		if length(sz) == 0  # scalar
			push!(ex, :($v = $PARAM_SYM[ $pos ]) )
			pos += 1
		elseif length(sz) == 1  # vector
			r = pos:(pos+sz[1]-1)
			push!(ex, :($v = $PARAM_SYM[ $(Expr(:(:), pos, pos+sz[1]-1)) ]) )
			pos += sz[1]
		else # matrix case  (needs a reshape)
			r = pos:(pos+prod(sz)-1)
			push!(ex, :($v = reshape($PARAM_SYM[ $(Expr(:(:), pos, pos+prod(sz)-1)) ], $(sz[1]), $(sz[2]))) )
			pos += prod(sz)
		end
	end
	ex
end

### creates mapping statements from model parameter variables to Vector{Float64}
# FIXME : using undocumented dprefix function of ReverseDiffSource (should be replaced)
function var2vec(;init...)
	ex = {}
	for (v,i) in init
		sz = size(i)
		if in(length(sz), [0,1]) # scalar or vector
			push!(ex, ReverseDiffSource.dprefix(v))
		else # matrix case  (needs a reshape)
			push!(ex, :( vec($(ReverseDiffSource.dprefix(v))) ) )
		end
	end
	Expr(:vcat, ex...)
end

### returns parameter info : total size, vector <-> model parameter map, inital values vector
function modelVars(;init...)
	# init = [(:x, 3.)]
    pars = Dict{Symbol, NTuple{2}}()
    pos = 1
    vi = Float64[]

    for (par, def) in init  # par, def = init[1]
    	isa(def, Real) || isa(def, AbstractVector) || isa(def, AbstractMatrix) ||
    		error("unsupported type for parameter $(par)")

        pars[par] = (pos, size(def))
        pos += length(def)
        vi = [vi, float64([def...])]
    end
    (pos-1, pars, vi)
end