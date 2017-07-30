IntegerVector{I<:Integer} = Vector{I}
RealVector{N<:Real} = Vector{N}
RealMatrix{N<:Real} = Matrix{N}

RealLowerTriangular{T, S} = LowerTriangular{T,S} where {T<:Real, S<:AbstractMatrix{T}}
RealLowerTriangular(m) = LowerTriangular(m)

RealNormal{N<:Real} = Normal{N}
MultivariateGMM{D<:MvNormal} = MultivariateMixture{Continuous, D}

multivecs{T}(::Type{T}, n::Int) = [T[] for _ =1:n]
