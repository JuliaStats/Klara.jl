#########################################################################
#    Helper functions for derivation tests
#########################################################################

# FIXME : remove calls to getSymbols and SubstSymbols

## error thresholds
DIFF_DELTA = 1e-9
ERROR_THRESHOLD = 2e-2

good_enough(x,y) = isfinite(x) ? (abs(x-y) / max(ERROR_THRESHOLD, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
good_enough(t::Tuple) = good_enough(t[1], t[2])

##  gradient check by comparing numerical gradient to automated gradient
function deriv1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) 

	nx = length(x0)  

	ex2 = :( $(MCMC.ACC_SYM) += $ex)	     
	fex2 = MCMC.generateModelFunction(ex2, gradient=true, debug=true, x=x0) 
	myf, dummy = MCMC.generateModelFunction(ex2, gradient=true, x=x0)

	pars0 = vec([x0])
	l0, grad0 = myf(pars0)  
	gradn = Array(Float64,nx)
	for i in 1:nx 
		l, grad = myf( Float64[ pars0[j] + (j==i)*DIFF_DELTA for j in 1:nx] )  
		gradn[i] = (l-l0)/DIFF_DELTA
	end

	if ! all(good_enough, zip([grad0], [gradn]))
		println("Gradient false for $ex at x=$x0, expected $(round(gradn,5)), got $(round(grad0,5))")
		println()
		println( fex2 )
		error()
	end
end

##  tests derivation on all parameters, for all combinations of arguments dimension
macro test_combin(func::Expr, constraints...)
	constraints = collect(constraints) 
	parnames = collect(MCMC.ReverseDiffSource.getSymbols(func))
	# println("parnames : $parnames")
	# dump(constraints)
	#  args to derive against
	dargs = filter(ex->isa(ex,Symbol), constraints)
	length(dargs)==0 && (dargs = parnames)
	@assert all(map(x->in(x,parnames), dargs)) "some of specified derivation args $(dargs) not found in tested expression $func"
	# println("dargs : $dargs")

	#  transformations on args to have valid calls
	trans = filter(ex->isa(ex, Expr) && ex.head == :(->), constraints) 
	# println("trans : $trans")

	#  dimension validity rules
	rules = filter(ex->isa(ex, Expr) && ex.head != :(->), constraints) 
	# println("rules : $rules")

	arity = length(parnames)
	par = [ symbol("arg$i") for i in 1:arity]
	
	# try each arg dim combination
	combin = [ [:v0ref, :v1ref, :v2ref][1+ ifloor((i-1) / 3^(j-1)) % 3] for i in 1:3^arity, j in 1:arity]
	for ic in 1:size(combin,1)  # try each arg dim in combin
		# create variables
		for i in 1:arity  
			eval(:( $(par[i]) = copy($(combin[ic,i]))) )
		end

		# reject combination if one of rules fails
		if all( [ eval( MCMC.ReverseDiffSource.substSymbols(r, Dict(parnames, par))) for r in rules] )
			# println("## combin = $(combin[ic,:])")
			# apply transformations on args
			for t in trans
				pos = find(parnames .== t.args[1]) # find the arg the rules applies to 
				@assert length(pos)>0 "arg of transfo ($t.args[1]) not found among $parnames"
				vn = symbol("arg$(pos[1])")
				eval(:( $vn = map($t, $vn)))
			end

			# try derivation for each allowed argument
			for p in dargs 
				tpar = copy(par)
				tpar[p .== parnames] = :x  # replace tested args with parameter symbol :x for deriv1 testing func
				fex = MCMC.ReverseDiffSource.substSymbols(func, Dict( parnames, tpar))
				# println("##+## $fex  $(Dict( parnames, tpar))")
				x0 = eval(par[p .== parnames][1])  # set x0 for deriv 1
				# println("##-## $fex  ##-##   $x0")
				deriv1(fex, x0) 
			end
		end
	end

end

