IntegerVector{I<:Integer} = Vector{I}
RealVector{N<:Real} = Vector{N}
RealMatrix{N<:Real} = Matrix{N}

RealLowerTriangular{N, M} = LowerTriangular{N, M} where {N<:Real, M<:AbstractMatrix{N}}
RealLowerTriangular(m) = LowerTriangular(m)

RealPair{F<:Real, S<:Real} = Pair{F, S}
RealPairVector{P<:RealPair} = Vector{P}
# RealPairVector{N, P} = Vector{P} where {N<:Real, P<:RealPair{N}}

FunctionVector{F<:Function} = Vector{F}

RealNormal{N<:Real} = Normal{N}
MultivariateGMM{D<:MvNormal} = MultivariateMixture{Continuous, D}

multivecs{T}(::Type{T}, n::Int) = [T[] for _ =1:n]
