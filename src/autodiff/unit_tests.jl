#################################################################
#    Model Expression Parsing and Autodiff Unit tests
#################################################################

using Base.Test

include("p:/Documents/julia/MCMC.jl.fredo/src/autodiff/Autodiff.jl")
testedmod = Autodiff

@test testedmod.isSymbol(:a)            == true
@test testedmod.isSymbol(:(a[1]))       == false
@test testedmod.isSymbol(:(a.b))        == false
@test testedmod.isSymbol(:(exp(a)))     == false

@test testedmod.isRef(:a)            == false
@test testedmod.isRef(:(a[1]))       == true
@test testedmod.isRef(:(a[x]))       == true
@test testedmod.isRef(:(a.b))        == false
@test testedmod.isRef(:(a.b[end]))   == false
@test testedmod.isRef(:(a[end].b))   == false
@test testedmod.isRef(:(exp(a)))     == false

@test testedmod.isDot(:a)           == false
@test testedmod.isDot(:(a[1]))      == false
@test testedmod.isDot(:(a[x]))      == false
@test testedmod.isDot(:(a.b))       == true
@test testedmod.isDot(:(a.b[end]))  == false
@test testedmod.isDot(:(a[end].b))  == false
@test testedmod.isDot(:(exp(a)))    == false


@test testedmod.dprefix("coucou")            == :_dcoucou
@test testedmod.dprefix(:tr)                 == :_dtr
@test testedmod.dprefix(:(b[2]))             == :(_db[2])
@test testedmod.dprefix(:(foo.bar))          == :(_dfoo.bar)
@test testedmod.dprefix(:(foo.bar.baz))      == :(_dfoo.bar.baz)
@test testedmod.dprefix(:(foo.bar[2].baz))   == :(_dfoo.bar[2].baz)
@test testedmod.dprefix(:(foo[3].bar))       == :(_dfoo[3].bar)


@test testedmod.getSymbols(:abc)             == Set(:abc)
@test testedmod.getSymbols([:abc, :foo])     == Set(:abc, :foo)
@test testedmod.getSymbols(:(a+b))           == Set(:b, :a)
@test testedmod.getSymbols(:(x=a+b ; z = y*x)) == Set(:b, :a, :x, :y, :z)
@test testedmod.getSymbols(:(sin(x)))        == Set(:x)
@test testedmod.getSymbols(:(log( exp( max(a,b))))) == Set(:a, :b)
@test testedmod.getSymbols(:(foo.bar))       == Set(:foo)
@test testedmod.getSymbols(:(foo[x]))        == Set(:foo, :x)
@test testedmod.getSymbols(:(foo[1,2,a]))    == Set(:foo, :a)
@test testedmod.getSymbols(:(foo[end,:,a]))  == Set(:foo, :a)
@test testedmod.getSymbols(:(foo[end,:,a]))  == Set(:foo, :a)
@test testedmod.getSymbols(:([ exp(i) for i in 1:10])) == Set(:i)
@test testedmod.getSymbols(:(if x >= 4 & y < 5 || w == 3; return y <= x ; end)) == Set(:x, :y, :w)


## variable subsitution function
smap = Dict()
@test testedmod.substSymbols(:(a+b), smap)             == :(a+b)

smap = {:a => :x, :b => :y}
@test testedmod.substSymbols(:a, smap)                 == :x
@test testedmod.substSymbols(:(a+b), smap)             == :(x+y)
@test testedmod.substSymbols(:(exp(a)), smap)          == :(exp(x))
@test testedmod.substSymbols(:(x=exp(a);y=log(c)), smap) == :(x=exp(x); y=log(c))
@test testedmod.substSymbols(:(a[z]), smap)            == :(x[z])
@test testedmod.substSymbols(:(z[a]), smap)            == :(z[x])
@test testedmod.substSymbols(:(a.z), smap)             == :(x.z)
@test testedmod.substSymbols(:(z.a), smap)             == :(z.a)     # note : no subst on field names
@test testedmod.substSymbols(:(z.a[x]), smap)          == :(z.a[x])  # note : no subst on field names
@test testedmod.substSymbols(:(z[x].a), smap)          == :(z[x].a)  # note : no subst on field names
@test testedmod.substSymbols(:(a[x].z), smap)          == :(x[x].z)  


## expression unfolding
macro unfold(ex)
	m = testedmod.ParsingStruct()
	m.source = ex
	testedmod.resetvar()  # needed to have a constant temporary var name generated
	testedmod.unfold!(m)
	m.exprs
end

@test (@unfold a = b+6)           == [:(a=b+6)]
@test (@unfold (sin(y);a=3))      == [:(sin(y)), :(a=3)]
@test (@unfold a[4] = b+6)        == [:(a[4]=b+6)]
@test (@unfold a += b+6)          == [:($(symbol("tmp#1"))=b+6), :(a = +(a,$(symbol("tmp#1"))))]
@test (@unfold a -= b+6)          == [:($(symbol("tmp#1"))=b+6), :(a = -(a,$(symbol("tmp#1"))))]
@test (@unfold a *= b+6)          == [:($(symbol("tmp#1"))=b+6), :(a = *(a,$(symbol("tmp#1"))))]
@test (@unfold b = a')            == [:(b=transpose(a))]
@test (@unfold a = [1,2])         == [:(a=vcat(1,2))]
@test_throws (@unfold a.b = 3.) 
@test (@unfold a = b.f)           == [:(a=b.f)]
@test (@unfold a = b.f[i])        == [:(a=b.f[i])]
@test (@unfold a = b[j].f[i])     == [:(a=b[j].f[i])]




